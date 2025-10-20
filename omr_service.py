import cv2
import numpy as np
import pytesseract
import os

def analyze_omr_sheet(image_path):
    """
    OMR-style javob varaqasini tahlil qiladi.
    :param image_path: Rasm (jpg/png) fayl yo'li
    :return: {'variant_id': str, 'answers': [int, ...]}
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
    num_columns = 3
    questions_per_col = num_questions // num_columns
    width, height = 2480, 3508
    margin = 100
    cell_size = 40
    gap_x = 20
    gap_y = 20
    start_y = 250
    start_x = margin
    col_width = (width - 2 * margin) // num_columns
    variant_labels = ['A', 'B', 'C', 'D']

    # Faqat bo'yalgan variantlarni aniqlash
    marked = dict()
    for col in range(num_columns):
        x_base = start_x + col * col_width
        for row in range(questions_per_col):
            y = start_y + row * (cell_size + gap_y)
            qnum = col * questions_per_col + row + 1
            found = False
            for v in range(num_variants):
                vx = x_base + 80 + v * (cell_size + gap_x)
                vy = y
                # Doira ichidan kichikroq ROI ajratamiz
                roi = thresh[vy+5:vy+cell_size-5, vx+5:vx+cell_size-5]
                total = roi.size
                black = np.count_nonzero(roi)
                fill_ratio = black / total
                # 0.6 dan katta bo'lsa, bo'yalgan deb hisoblaymiz
                if fill_ratio > 0.6:
                    marked[str(qnum)] = variant_labels[v]
                    found = True
            # Agar hech biri bo'yalanmagan bo'lsa, natijaga qo'shilmaydi (null qaytmaydi, shunchaki yo'q)

    # Variant ID joyini aniqlash (masalan, yuqoridagi maxsus joy)
    # Bu joyni rasm formatiga moslab o'zgartiring
    # OCR uchun joyni kengaytirish va faqat raqamlarni ajratish
    variant_roi = img[100:180, 100:700]  # "Varaqa ID" yozuvi joylashgan joy
    ocr_text = pytesseract.image_to_string(variant_roi, config='--psm 7').strip()
    import re
    match = re.search(r'(\d{6,})', ocr_text)
    variant_id = match.group(1) if match else ''

    return {'variant_id': variant_id, 'answers': marked}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Foydalanish: python omr_service.py <scan.jpg>")
        exit(1)
    image_path = sys.argv[1]
    result = analyze_omr_sheet(image_path)
    print("Natija:", result)
