#!/usr/bin/env python3
"""
Answer Sheet Analyzer - Javob varag'ini tahlil qilish
To'ldirilgan doiralarni aniqlash va javoblarni chiqarish
"""
import cv2
import numpy as np
from PIL import Image
import os
import json

def analyze_answer_sheet(image_path):
    """
    Javob varag'ini tahlil qilib, belgilangan javoblarni aniqlaydi
    """
    if not os.path.exists(image_path):
        print(f"❌ Xato: {image_path} fayl topilmadi!")
        return None

    # Rasmni yuklash
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Xato: Rasmni o'qib bo'lmadi!")
        return None

    # Original o'lchamlarni saqlash
    original = img.copy()
    height, width = img.shape[:2]

    print(f"\n{'='*70}")
    print(f"📄 JAVOB VARAQASI TAHLILI")
    print(f"{'='*70}")
    print(f"📁 Fayl: {os.path.basename(image_path)}")
    print(f"📐 O'lcham: {width}x{height} piksel")

    # Grayscale ga o'tkazish
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold - matn va chiziqlarni ajratish
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Doiralarni topish - Hough Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=8,
        maxRadius=25
    )

    if circles is None:
        print("⚠️  Hech qanday doira topilmadi!")
        return None

    circles = np.uint16(np.around(circles))

    print(f"\n🔍 Topilgan doiralar soni: {len(circles[0])}")

    # Doiralarni tahlil qilish
    filled_circles = []
    all_circles = []

    for i, (x, y, r) in enumerate(circles[0, :]):
        # 1. Doira ichidagi o'rtacha brightness (grayscale)
        mask_full = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask_full, (x, y), int(r * 0.7), 255, -1)
        mean_val = cv2.mean(gray, mask=mask_full)[0]

        # 2. Doira markazidagi qora piksellar nisbati (grayscale)
        mask_center = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask_center, (x, y), int(r * 0.5), 255, -1)
        center_pixels = gray[mask_center > 0]

        # Qora piksellar soni (threshold < 100)
        dark_pixels_count = np.sum(center_pixels < 100) if len(center_pixels) > 0 else 0
        dark_ratio_gray = dark_pixels_count / len(center_pixels) if len(center_pixels) > 0 else 0

        # 3. Rangli piksellarni tekshirish (HSV orqali)
        # BGR rangli rasmdan doira ichidagi qismni olish
        center_pixels_bgr = img[mask_center > 0]

        # HSV ga o'tkazish va saturation (to'yinganlik) tekshirish
        if len(center_pixels_bgr) > 0:
            # Doira ichidagi rang ma'lumotlarini olish
            roi = cv2.bitwise_and(img, img, mask=mask_center)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # To'q rangli piksellar (saturation yuqori)
            # H: har qanday rang, S: 50 dan yuqori (to'q), V: 30 dan yuqori (juda qora emas)
            saturated_pixels = roi_hsv[mask_center > 0]
            high_saturation = np.sum(saturated_pixels[:, 1] > 50) if len(saturated_pixels) > 0 else 0
            saturation_ratio = high_saturation / len(saturated_pixels) if len(saturated_pixels) > 0 else 0
        else:
            saturation_ratio = 0

        # 4. Standart og'ish (kontrast)
        std_val = np.std(center_pixels) if len(center_pixels) > 0 else 0

        # To'ldirilganligini aniqlash:
        # Variant 1: Qora rang bilan bo'yalgan (dark_ratio_gray > 0.6)
        # Variant 2: To'q rangli bo'yalgan (saturation_ratio > 0.5)
        # Variant 3: Brightness past (mean_val < 120) - juda to'q
        is_filled = (dark_ratio_gray > 0.6 and mean_val < 180) or \
                    (saturation_ratio > 0.5 and mean_val < 220) or \
                    (mean_val < 120)

        circle_data = {
            'x': int(x),
            'y': int(y),
            'radius': int(r),
            'brightness': round(mean_val, 2),
            'std_dev': round(std_val, 2),
            'dark_ratio': round(dark_ratio_gray, 3),
            'saturation_ratio': round(saturation_ratio, 3),
            'filled': is_filled
        }

        all_circles.append(circle_data)

        if is_filled:
            filled_circles.append(circle_data)

    print(f"✅ To'ldirilgan doiralar: {len(filled_circles)}")

    # Barcha doiralarni Y pozitsiyasi bo'yicha saralash (yuqoridan pastga)
    all_circles.sort(key=lambda c: (c['y'], c['x']))

    # Y pozitsiyasiga qarab qatorlarga ajratish (clustering)
    # Yaqin Y qiymatlaridagi doiralarni bir qatorda deb hisoblash
    rows = []
    current_row = []
    prev_y = None

    for circle in all_circles:
        if prev_y is None or abs(circle['y'] - prev_y) <= 5:
            # Bir qatorda (5 piksel farq chegarasi)
            current_row.append(circle)
        else:
            # Yangi qator
            if len(current_row) >= 4:
                rows.append(sorted(current_row, key=lambda c: c['x'])[:4])  # Faqat birinchi 4 tasini (A,B,C,D)
            current_row = [circle]
        prev_y = circle['y']

    # Oxirgi qatorni ham qo'shish
    if len(current_row) >= 4:
        rows.append(sorted(current_row, key=lambda c: c['x'])[:4])

    print(f"🔢 Topilgan javob qatorlari (savollar): {len(rows)}")

    # Javoblarni aniqlash
    answers = {}
    options = ['A', 'B', 'C', 'D']

    print(f"\n{'='*70}")
    print(f"📝 JAVOBLAR RO'YXATI")
    print(f"{'='*70}")

    result_visualization = original.copy()

    # Har bir qatorni tahlil qilish
    for question_num, row_circles in enumerate(rows, start=1):
        selected_answer = None

        # To'ldirilgan doirani topish
        for idx, circle in enumerate(row_circles):
            if circle['filled']:
                selected_answer = options[idx]
                # Vizualizatsiya
                cv2.circle(result_visualization,
                         (circle['x'], circle['y']),
                         circle['radius'], (0, 255, 0), 3)
                cv2.putText(result_visualization,
                          f"{question_num}={selected_answer}",
                          (circle['x'] + 25, circle['y']),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                break

        if selected_answer:
            answers[question_num] = selected_answer
            print(f"Savol {question_num:3d}: {selected_answer}")
        else:
            # Faqat birinchi 20 ta va oxirgi 5 ta savolni chop etish
            if question_num <= 20 or question_num > len(rows) - 5:
                print(f"Savol {question_num:3d}: (javob yo'q)")

    # Agar ko'p javob yo'q bo'lsa, o'rtadagilarni ham ko'rsatish
    if question_num > 20 and question_num <= len(rows) - 5:
        skipped = len(rows) - 25
        if skipped > 0:
            print(f"... va yana {skipped} ta savol (javob yo'q)")

    question_total = len(rows)

    # To'ldirilmagan doiralarni ko'k rangda belgilash
    for circle in all_circles:
        if not circle['filled']:
            cv2.circle(result_visualization,
                     (circle['x'], circle['y']),
                     circle['radius'], (255, 0, 0), 1)

    # Natijalarni saqlash
    output_dir = os.path.join(os.path.dirname(image_path), 'analyzed_output')
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Vizualizatsiya rasmini saqlash
    result_path = os.path.join(output_dir, f"{base_name}_answers.png")
    cv2.imwrite(result_path, result_visualization)

    # JSON formatda saqlash
    json_path = os.path.join(output_dir, f"{base_name}_answers.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)

    # Tekst formatda saqlash
    txt_path = os.path.join(output_dir, f"{base_name}_answers.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("JAVOBLAR RO'YXATI\n")
        f.write("="*50 + "\n\n")
        for q_num, answer in sorted(answers.items()):
            f.write(f"Savol {q_num}: {answer}\n")

    print(f"\n{'='*70}")
    print(f"💾 NATIJALAR SAQLANDI")
    print(f"{'='*70}")
    print(f"🖼️  Vizual natija: {result_path}")
    print(f"📄 JSON format: {json_path}")
    print(f"📝 Text format: {txt_path}")

    print(f"\n{'='*70}")
    print(f"📊 XULOSA")
    print(f"{'='*70}")
    print(f"Jami savollar: {question_num}")
    print(f"Javob berilgan: {len(answers)}")
    print(f"Javob berilmagan: {question_num - len(answers)}")
    print(f"{'='*70}\n")

    return {
        'answers': answers,
        'total_questions': question_num,
        'answered': len(answers),
        'output_path': result_path,
        'json_path': json_path,
        'txt_path': txt_path
    }

def main():
    import sys

    # Faylni argument sifatida olish yoki standart fayldan foydalanish
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Standart fayl yo'li
        image_path = "/home/behzod/Downloads/Javoblar Varagi — Kimyo (35 ta savol)-1.png"

    if not os.path.exists(image_path):
        print(f"⚠️  Fayl topilmadi: {image_path}")
        print("\n💡 Foydalanish: python answer_sheet_analyzer.py <rasm_fayl_yo'li>")
        print("Yoki to'g'ri fayl yo'lini kiriting:")
        image_path = input("Fayl yo'li: ").strip()

        if not os.path.exists(image_path):
            print("❌ Fayl hali ham topilmadi!")
            return

    # Tahlil qilish
    result = analyze_answer_sheet(image_path)

    if result:
        print("✅ Tahlil muvaffaqiyatli yakunlandi!")

        # Javoblarni ekranga chiqarish
        if result['answers']:
            print("\n📋 QISQACHA NATIJA:")
            answers_str = ", ".join([f"{k}={v}" for k, v in sorted(result['answers'].items())])
            print(f"   {answers_str}")

            # JSON formatda ham chiqarish
            print("\n📄 JSON FORMAT:")
            import json
            print(json.dumps(result['answers'], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
