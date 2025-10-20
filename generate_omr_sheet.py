from PIL import Image, ImageDraw, ImageFont
import os

def generate_omr_sheet(sheet_path, num_questions=108, num_variants=4, sheet_id="00000001"):
    # OMR varaq o'lchami (A4: 2480x3508 px, 300dpi)
    width, height = 2480, 3508
    margin = 100
    cell_size = 40
    gap_x = 20
    gap_y = 20
    start_y = 250
    start_x = margin
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Linux uchun
    font = ImageFont.truetype(font_path, 36)
    small_font = ImageFont.truetype(font_path, 28)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Sheet ID faqat bitta kichik joyda
    draw.text((margin, 50), f"Javoblar varaqi", font=font, fill="black")

    # Katta joylashuv: savollar va variantlar A4 ni to'ldiradi
    num_columns = 3
    questions_per_col = num_questions // num_columns
    col_width = (width - 2 * margin) // num_columns
    big_cell_size = 60
    big_gap_x = 32
    big_gap_y = 32
    big_font = ImageFont.truetype(font_path, 48)
    big_small_font = ImageFont.truetype(font_path, 38)
    for col in range(num_columns):
        for row in range(questions_per_col):
            i = col * questions_per_col + row
            y = start_y + row * (big_cell_size + big_gap_y)
            x_base = start_x + col * col_width
            draw.text((x_base, y), f"{i+1:03}", font=big_small_font, fill="black")
            for v in range(num_variants):
                x = x_base + 120 + v * (big_cell_size + big_gap_x)
                # Katta doira (variant)
                draw.ellipse((x, y, x+big_cell_size, y+big_cell_size), outline="black", width=4)
                draw.text((x+big_cell_size//2-14, y+big_cell_size//2-24), chr(65+v), font=big_small_font, fill="black")

    # Ism, familiya, variant ID grid yo'q

    # Saqlash
    img.save(sheet_path)
    print(f"OMR varaqi saqlandi: {sheet_path}")

if __name__ == "__main__":
    generate_omr_sheet("omr_example_108.png", num_questions=108, sheet_id="12345678")
