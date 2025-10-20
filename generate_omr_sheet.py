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

    # Sheet ID va nomer
    draw.text((margin, 50), f"Javoblar varaqi", font=font, fill="black")
    draw.text((margin, 120), f"Varaqa ID: {sheet_id}", font=small_font, fill="black")

    # 3 ta ustunli joylashuv
    num_columns = 3
    questions_per_col = num_questions // num_columns
    col_width = (width - 2 * margin) // num_columns
    for col in range(num_columns):
        for row in range(questions_per_col):
            i = col * questions_per_col + row
            y = start_y + row * (cell_size + gap_y)
            x_base = start_x + col * col_width
            draw.text((x_base, y), f"{i+1:03}", font=small_font, fill="black")
            for v in range(num_variants):
                x = x_base + 80 + v * (cell_size + gap_x)
                # Doira (variant)
                draw.ellipse((x, y, x+cell_size, y+cell_size), outline="black", width=2)
                draw.text((x+cell_size//2-10, y+cell_size//2-18), chr(65+v), font=small_font, fill="black")

    # Pastki qismga ism va familiya uchun faqat tekstli joy
    name_y = height - 250
    label_font = ImageFont.truetype(font_path, 32)
    draw.text((margin, name_y), "Ism: ______________________________________", font=label_font, fill="black")
    surname_y = name_y + 60
    draw.text((margin, surname_y), "Familiya: __________________________________", font=label_font, fill="black")

    # Saqlash
    img.save(sheet_path)
    print(f"OMR varaqi saqlandi: {sheet_path}")

if __name__ == "__main__":
    generate_omr_sheet("omr_example_108.png", num_questions=108, sheet_id="12345678")
