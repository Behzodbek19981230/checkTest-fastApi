import os
from pathlib import Path

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _warp_to_a4(gray: np.ndarray, out_w: int | None = 2480, out_h: int | None = 3508) -> np.ndarray:
    """Find page contour and warp to canonical A4 raster (approx 300dpi).

    If contour detection fails, falls back to resizing.
    """
    if gray is None or gray.size == 0:
        raise ValueError("Empty image")

    if out_w is None:
        out_w = int(gray.shape[1])
    if out_h is None:
        # Keep A4 aspect ratio
        out_h = int(round(float(out_w) * (842.0 / 595.0)))

    # Edge-based page detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    page = None
    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            page = approx.reshape(4, 2).astype("float32")
            break

    if page is None:
        return cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)

    rect = _order_points(page)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (out_w, out_h))


def _threshold_inv(gray: np.ndarray) -> np.ndarray:
    """Return an inverted binary image.

    Tries adaptive threshold first; falls back to Otsu if needed.
    """
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    # Otsu fallback (sometimes better for clean printed templates)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Pick the one that yields a "reasonable" ink ratio.
    a_ratio = float(np.count_nonzero(adaptive)) / float(adaptive.size)
    o_ratio = float(np.count_nonzero(otsu)) / float(otsu.size)

    # In practice, for clean printed forms (and especially PDF renders),
    # Otsu is usually less noisy. Prefer it when it produces noticeably less ink.
    if o_ratio < a_ratio * 0.90 and 0.02 < o_ratio < 0.20:
        return otsu
    return adaptive


def _find_bubbles(thresh: np.ndarray, min_area: int, max_area: int) -> list[tuple[int, int, int]]:
    """Return list of (cx, cy, r) for circle-like contours."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles: list[tuple[int, int, int]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ar = w / float(h)
        if ar < 0.75 or ar > 1.25:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.55:
            continue
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        r = int(round((w + h) / 4))
        bubbles.append((cx, cy, r))
    return bubbles


def _find_bubbles_hough(gray: np.ndarray) -> list[tuple[int, int, int]]:
    """Fallback circle detector for clean printed forms."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=28,
        param1=120,
        param2=22,
        minRadius=10,
        maxRadius=40,
    )
    out: list[tuple[int, int, int]] = []
    if circles is None:
        return out
    circles = np.round(circles[0, :]).astype("int")
    for x, y, r in circles:
        out.append((int(x), int(y), int(r)))
    return out


def _group_by_y(points: list[tuple[int, int, int]], tol: int) -> list[list[tuple[int, int, int]]]:
    pts = sorted(points, key=lambda t: t[1])
    groups: list[list[tuple[int, int, int]]] = []
    for p in pts:
        if not groups:
            groups.append([p])
            continue
        if abs(p[1] - groups[-1][0][1]) <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


def _fill_ratio(thresh: np.ndarray, cx: int, cy: int, r: int) -> float:
    # Sample a slightly smaller square inside the bubble to avoid border stroke.
    # Printed bubbles have a dark ring; we must avoid counting that as a fill.
    pad = max(3, int(r * 0.65))
    x1 = max(0, cx - r + pad)
    y1 = max(0, cy - r + pad)
    x2 = min(thresh.shape[1], cx + r - pad)
    y2 = min(thresh.shape[0], cy + r - pad)
    roi = thresh[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    black = int(np.count_nonzero(roi))
    return black / float(roi.size)


def _fill_score_gray(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Return darkness score in [0..1] from grayscale (higher => more filled)."""
    pad = max(3, int(r * 0.55))
    x1 = max(0, cx - r + pad)
    y1 = max(0, cy - r + pad)
    x2 = min(gray.shape[1], cx + r - pad)
    y2 = min(gray.shape[0], cy + r - pad)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    mean = float(np.mean(roi))
    return max(0.0, min(1.0, (255.0 - mean) / 255.0))


def _fill_score_delta(scan_gray: np.ndarray, ref_gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    """Return additional darkness compared to reference template in [0..1]."""
    pad = max(3, int(r * 0.55))
    x1 = max(0, cx - r + pad)
    y1 = max(0, cy - r + pad)
    x2 = min(scan_gray.shape[1], cx + r - pad)
    y2 = min(scan_gray.shape[0], cy + r - pad)
    roi_s = scan_gray[y1:y2, x1:x2]
    roi_r = ref_gray[y1:y2, x1:x2]
    if roi_s.size == 0 or roi_r.size == 0:
        return 0.0
    mean_s = float(np.mean(roi_s))
    mean_r = float(np.mean(roi_r))
    # If scan is darker than reference => positive score
    return max(0.0, min(1.0, (mean_r - mean_s) / 255.0))


def _default_titul_template_path() -> Path | None:
    # Default: look for Titul_variant.png at workspace root (one level above checkTest-fastApi)
    here = Path(__file__).resolve()
    root = here.parent.parent
    p = root / "Titul_variant.png"
    if p.exists():
        return p
    return None


def _load_template_warped_gray(out_w: int, out_h: int) -> np.ndarray | None:
    """Load template image from env or default location and warp/resize to target."""
    env_path = (os.getenv("TITUL_TEMPLATE_IMAGE") or "").strip()
    path = Path(env_path) if env_path else _default_titul_template_path()
    if not path or not path.exists():
        return None
    ref = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        return None
    return _warp_to_a4(ref, out_w=out_w, out_h=out_h)


def analyze_titul_sheet(image_path: str, variant_digits: int = 5) -> dict:
    """Analyze scanned Titul-style OMR sheet.

    Expects 3 answer columns (1-30, 31-60, 61-90) with A-D bubbles,
    and a top-right Variant grid (10 rows x variant_digits columns).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Rasm topilmadi: {image_path}")

    # Avoid upscaling (introduces threshold noise); keep the scan's native resolution.
    out_w = int(img.shape[1])
    out_h = int(round(out_w * (842.0 / 595.0)))
    warped = _warp_to_a4(img, out_w=out_w, out_h=out_h)
    thresh = _threshold_inv(warped)

    ref_warped = _load_template_warped_gray(out_w=out_w, out_h=out_h)
    use_ref = ref_warped is not None

    H, W = thresh.shape[:2]
    # Find bubbles (tuned for ~300dpi)
    # First try contour-based bubbles, then HoughCircles fallback.
    # (Printed circles can be thin and broken in thresholded images.)
    bubbles = _find_bubbles(thresh, min_area=80, max_area=8000)
    if not bubbles:
        bubbles = _find_bubbles_hough(warped)
    if not bubbles:
        raise ValueError("Bubbles topilmadi (scan sifati yoki threshold muammo bo'lishi mumkin)")

    # Split bubbles into answer area vs variant area by x position.
    ans_bubbles = [b for b in bubbles if b[0] < int(W * 0.62) and int(H * 0.30) < b[1] < int(H * 0.78)]
    var_bubbles = [b for b in bubbles if b[0] > int(W * 0.70) and int(H * 0.32) < b[1] < int(H * 0.55)]

    # --- Variant (10 rows x variant_digits cols) ---
    variant_id = "_" * int(variant_digits)
    if var_bubbles:
        # Group by rows (digit values 0..9)
        rs = [b[2] for b in var_bubbles]
        tol = max(10, int(np.median(rs) * 0.9))
        row_groups = _group_by_y(var_bubbles, tol=tol)
        # Keep the 10 densest rows (some scans may have extra noise)
        row_groups = sorted(row_groups, key=lambda g: (g[0][1], -len(g)))
        if len(row_groups) >= 10:
            # choose 10 rows closest together by y-span
            # simplest: take first 10
            row_groups = row_groups[:10]

        # Normalize each row to exactly variant_digits bubbles by x order (take closest)
        row_groups = [sorted(g, key=lambda t: t[0]) for g in row_groups]
        # Build fill matrix [rows][cols]
        fills = []
        for g in row_groups:
            if len(g) < variant_digits:
                continue
            # If more than needed, keep the right-most set (variant area is on the right)
            if len(g) > variant_digits:
                g = g[-int(variant_digits):]
            if use_ref:
                fills.append([_fill_score_delta(warped, ref_warped, cx, cy, r) for cx, cy, r in g])
            else:
                fills.append([_fill_score_gray(warped, cx, cy, r) for cx, cy, r in g])

        if len(fills) >= 10:
            fills = fills[:10]
            digits_out = []
            for col in range(int(variant_digits)):
                col_vals = [fills[row][col] for row in range(10)]
                best_row = int(np.argmax(col_vals))
                # Must be confidently above the others (avoid false positives from bubble rings)
                sorted_vals = sorted(col_vals, reverse=True)
                best_val = col_vals[best_row]
                second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
                if best_val > 0.10 and (best_val - second_val) > 0.03:
                    digits_out.append(str(best_row))
                else:
                    digits_out.append("_")
            variant_id = "".join(digits_out)

    # --- Answers (3 columns x 30 rows x 4 options) ---
    answers: dict[str, str] = {}
    if ans_bubbles:
        rs = [b[2] for b in ans_bubbles]
        tol = max(10, int(np.median(rs) * 0.9))
        rows = _group_by_y(ans_bubbles, tol=tol)
        # Keep rows that look like answer rows (expect ~12 bubbles per row)
        rows = [sorted(g, key=lambda t: t[0]) for g in rows if len(g) >= 8]
        rows = sorted(rows, key=lambda g: g[0][1])
        # Limit to 30 rows if scan contains extras
        if len(rows) > 30:
            rows = rows[:30]

        labels = ["A", "B", "C", "D"]
        for row_i, g in enumerate(rows):
            # Split by two biggest x gaps into 3 columns
            xs = [t[0] for t in g]
            gaps = []
            for i in range(len(xs) - 1):
                gaps.append((xs[i + 1] - xs[i], i))
            gaps = sorted(gaps, reverse=True)[:2]
            cuts = sorted([idx + 1 for _, idx in gaps])
            parts = []
            start = 0
            for c in cuts + [len(g)]:
                parts.append(g[start:c])
                start = c
            # We expect 3 parts; if not, skip
            if len(parts) != 3:
                continue

            qnums = [row_i + 1, row_i + 31, row_i + 61]
            for col_i, part in enumerate(parts):
                part = sorted(part, key=lambda t: t[0])
                if len(part) < 4:
                    continue
                # Choose 4 bubbles closest together (ignore stray)
                if len(part) > 4:
                    part = part[:4]
                if use_ref:
                    fills = [_fill_score_delta(warped, ref_warped, cx, cy, r) for cx, cy, r in part]
                else:
                    fills = [_fill_score_gray(warped, cx, cy, r) for cx, cy, r in part]
                best = int(np.argmax(fills))
                sorted_vals = sorted(fills, reverse=True)
                best_val = fills[best]
                second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
                if best_val > 0.10 and (best_val - second_val) > 0.03:
                    answers[str(qnums[col_i])] = labels[best]

    return {"variant_id": variant_id, "answers": answers}


def analyze_omr_sheet(image_path, code_digits: int = 5):
    """
    OMR-style javob varaqasini tahlil qiladi.
    :param image_path: Rasm (jpg/png) fayl yo'li
    :param code_digits: Test ID uzunligi (default: 5)
    :return: {'variant_id': str, 'answers': {qnum: 'A'|'B'|'C'|'D'}}
    """
    # Rasmni yuklash
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Rasm topilmadi: {image_path}")

    # Try Titul template first (new printed sheet). If it fails, fall back to legacy generator layout.
    try:
        return analyze_titul_sheet(image_path, variant_digits=code_digits)
    except Exception:
        # Legacy path continues below
        pass

    # Threshold (bo'yalgan joylarni ajratish)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

    # Konturlarni topish
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mapping uchun parametrlar (generate_omr_sheet.py bilan mos)
    num_questions = 108
    num_variants = 4
    num_columns = 4
    rows_per_col = int(np.ceil(num_questions / num_columns))
    width, height = 2480, 3508
    margin = 80
    bubble_size = 54
    gap_x = 22
    gap_y = 18
    # Must match generate_omr_sheet.py (moved down for header/info fields)
    start_y = 380
    start_x = margin
    col_width = (width - 2 * margin) // num_columns
    variant_labels = ['A', 'B', 'C', 'D']

    # Faqat bo'yalgan variantlarni aniqlash (yangi layout, 4 qator, katta doira)
    marked = dict()
    for col in range(num_columns):
        x_base = start_x + col * col_width
        for row in range(rows_per_col):
            y = start_y + row * (bubble_size + gap_y)
            qnum = col * rows_per_col + row + 1
            if qnum > num_questions:
                continue
            for v in range(num_variants):
                vx = x_base + 90 + v * (bubble_size + gap_x)
                vy = y
                roi = thresh[vy+8:vy+bubble_size-8, vx+8:vx+bubble_size-8]
                total = roi.size
                if total == 0:
                    continue
                black = np.count_nonzero(roi)
                fill_ratio = black / total
                if fill_ratio > 0.5:
                    marked[str(qnum)] = variant_labels[v]

    # Test ID: code_digits x 10 grid (pastki qismda, har bir qatorda 0-9 dan bittasi bo'yalgan)
    grid_left_x = margin
    grid_top_y = height - 760
    # Must match generate_omr_sheet.py (same as bubble_size)
    grid_cell_size = bubble_size
    grid_gap = 12
    variant_digits = []
    for row in range(int(code_digits)):
        y = grid_top_y + row * (grid_cell_size + grid_gap)
        found_digit = None
        for col in range(10):
            x = grid_left_x + col * (grid_cell_size + grid_gap)
            roi = thresh[y+8:y+grid_cell_size-8, x+8:x+grid_cell_size-8]
            total = roi.size
            if total == 0:
                continue
            black = np.count_nonzero(roi)
            fill_ratio = black / total
            if fill_ratio > 0.5:
                found_digit = str(col)
        if found_digit is not None:
            variant_digits.append(found_digit)
        else:
            variant_digits.append('_')  # Agar bo'yalmagan bo'lsa, aniqlanmagan
    variant_id = ''.join(variant_digits)

    return {'variant_id': variant_id, 'answers': marked}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Foydalanish: python omr_service.py <scan.jpg>")
        exit(1)
    image_path = sys.argv[1]
    result = analyze_omr_sheet(image_path)
    print("Natija:", result)
