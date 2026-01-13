import cv2
import numpy as np


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
