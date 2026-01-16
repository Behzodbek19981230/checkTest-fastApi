import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _pt_to_px(pt: float, scale: float) -> int:
    return int(round(pt * scale))


def _find_font() -> str | None:
    # Common on Ubuntu/Debian
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def update_titul_pdf(input_pdf: Path, output_pdf: Path, dpi: int = 300) -> None:
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF topilmadi: {input_pdf}")

    # Render PDF -> PNG (page 1)
    with tempfile.TemporaryDirectory(prefix="titul_edit_") as tmpdir:
        out_prefix = str(Path(tmpdir) / "page")
        subprocess.check_call(
            [
                "pdftoppm",
                "-f",
                "1",
                "-l",
                "1",
                "-png",
                "-r",
                str(dpi),
                str(input_pdf),
                out_prefix,
            ]
        )
        png_path = Path(f"{out_prefix}-1.png")
        if not png_path.exists():
            raise RuntimeError("pdftoppm PNG chiqarmadi")

        img = Image.open(png_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # PDF user space is points: 595x842 for A4 in this file
        scale_x = img.size[0] / 595.0
        scale_y = img.size[1] / 842.0
        # Should be identical; keep the safer mean.
        scale = (scale_x + scale_y) / 2.0

        # === 1) Remove top row labels (Majburiy fanlar / 1-Fan / 2-Fan) ===
        # Keep this tight to avoid erasing any separator/border lines.
        labels_left_x1_pt = 55.0
        labels_left_y1_pt = 236.0
        labels_left_x2_pt = 330.0
        labels_left_y2_pt = 262.0
        draw.rectangle(
            (
                _pt_to_px(labels_left_x1_pt, scale),
                _pt_to_px(labels_left_y1_pt, scale),
                _pt_to_px(labels_left_x2_pt, scale),
                _pt_to_px(labels_left_y2_pt, scale),
            ),
            fill="white",
        )

        # === 2) Remove "Kitob raqami" label + bubbles ===
        # Split into (a) label strip and (b) bubble grid area so we don't wipe header separators.
        kitob_label_x1_pt = 360.0
        kitob_label_y1_pt = 236.0
        kitob_label_x2_pt = 470.0
        kitob_label_y2_pt = 262.0
        draw.rectangle(
            (
                _pt_to_px(kitob_label_x1_pt, scale),
                _pt_to_px(kitob_label_y1_pt, scale),
                _pt_to_px(kitob_label_x2_pt, scale),
                _pt_to_px(kitob_label_y2_pt, scale),
            ),
            fill="white",
        )

        # Kitob digit columns are around x~364.5..442.5, digits y~296.9..419.3 (points)
        kitob_x1_pt = 352.0
        kitob_y1_pt = 270.0
        kitob_x2_pt = 470.0
        kitob_y2_pt = 436.0
        draw.rectangle(
            (
                _pt_to_px(kitob_x1_pt, scale),
                _pt_to_px(kitob_y1_pt, scale),
                _pt_to_px(kitob_x2_pt, scale),
                _pt_to_px(kitob_y2_pt, scale),
            ),
            fill="white",
        )

        # === 3) Convert "ID raqam" -> "Variant raqami" and make it 5-digit ===
        # Existing ID digit columns at x centers: 474.5, 487.5, 500.5, 513.5 (pt)
        # We add one more column at 461.5 (pt) by copying the first ID column slice.
        id_copy_y1_pt = 276.0
        id_copy_y2_pt = 436.0

        # Source slice boundaries: between col1 center (474.5) and col2 center (487.5)
        # midpoints => left=468.0, right=481.0
        src_x1_pt, src_x2_pt = 468.0, 481.0
        # Destination for new column centered at 461.5 => left=455.0, right=468.0
        dst_x1_pt, dst_x2_pt = 455.0, 468.0

        src_box = (
            _pt_to_px(src_x1_pt, scale),
            _pt_to_px(id_copy_y1_pt, scale),
            _pt_to_px(src_x2_pt, scale),
            _pt_to_px(id_copy_y2_pt, scale),
        )
        dst_box = (
            _pt_to_px(dst_x1_pt, scale),
            _pt_to_px(id_copy_y1_pt, scale),
            _pt_to_px(dst_x2_pt, scale),
            _pt_to_px(id_copy_y2_pt, scale),
        )

        # Clear destination then paste.
        # Note: with rounding, src and dst box sizes may differ by 1px.
        draw.rectangle(dst_box, fill="white")
        col_slice = img.crop(src_box)
        dst_w = max(1, dst_box[2] - dst_box[0])
        dst_h = max(1, dst_box[3] - dst_box[1])
        if col_slice.size != (dst_w, dst_h):
            col_slice = col_slice.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
        img.paste(col_slice, (dst_box[0], dst_box[1]))

        # Replace label area (cover old "ID raqam")
        label_x1_pt = 440.0
        label_y1_pt = 236.0
        label_x2_pt = 560.0
        label_y2_pt = 262.0
        label_box = (
            _pt_to_px(label_x1_pt, scale),
            _pt_to_px(label_y1_pt, scale),
            _pt_to_px(label_x2_pt, scale),
            _pt_to_px(label_y2_pt, scale),
        )
        draw.rectangle(label_box, fill="white")

        # Use non-bold font to match other small header labels.
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" if os.path.exists(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ) else _find_font()
        if font_path:
            font = ImageFont.truetype(font_path, size=int(round(8 * scale)))
        else:
            font = ImageFont.load_default()

        label_text = "Variant raqami"
        tx = _pt_to_px(452.0, scale)
        ty = _pt_to_px(244.0, scale)
        draw.text((tx, ty), label_text, fill=(0, 0, 0), font=font)

        # Export
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_png = output_pdf.with_suffix(".png")
        img.save(out_png)
        img.save(output_pdf, "PDF", resolution=float(dpi))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Titul.pdf answer-sheet template: remove 'Kitob raqami' and make 'Variant raqami' 5-digit. "
            "Outputs a new rasterized PDF."
        )
    )
    parser.add_argument(
        "--input",
        default="/home/behzod/Desktop/test/startup/Titul.pdf",
        help="Input PDF path",
    )
    parser.add_argument(
        "--output",
        default="/home/behzod/Desktop/test/startup/Titul_variant.pdf",
        help="Output PDF path",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI")
    args = parser.parse_args()

    update_titul_pdf(Path(args.input), Path(args.output), dpi=args.dpi)
    print(f"OK: {args.output}")


if __name__ == "__main__":
    main()
