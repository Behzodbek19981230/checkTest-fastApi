from PIL import Image, ImageDraw, ImageFont
import math
import os


def generate_omr_sheet(
    sheet_path: str,
    num_questions: int = 108,
    num_variants: int = 4,
    code_digits: int = 5,
):
    """Generate OMR sheet image.

    Layout assumptions are mirrored by `omr_service.analyze_omr_sheet`.
    """
    # OMR varaq o'lchami (A4: 2480x3508 px, 300dpi)
    width, height = 2480, 3508
    margin = 80
    start_x = margin
    # Leave space for header + personal info fields
    start_y = 380

    font_path = os.getenv(
        "OMR_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    )
    title_font = ImageFont.truetype(font_path, 52)
    label_font = ImageFont.truetype(font_path, 34)
    small_font = ImageFont.truetype(font_path, 26)
    tiny_font = ImageFont.truetype(font_path, 22)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Colors inspired by frontend title.html (keeps OMR area high-contrast)
    text_dark = (44, 62, 80)          # #2c3e50
    accent_blue = (52, 152, 219)      # #3498db
    border_light = (225, 229, 235)    # #e1e5eb
    field_border = (205, 214, 224)
    warning_bg = (255, 243, 205)      # #fff3cd
    warning_border = (255, 234, 167)  # #ffeaa7

    # Header block
    header_x1 = margin
    header_y1 = 40
    header_x2 = width - margin
    header_y2 = 330
    try:
        draw.rounded_rectangle(
            (header_x1, header_y1, header_x2, header_y2),
            radius=16,
            fill=(255, 255, 255),
            outline=border_light,
            width=3,
        )
    except Exception:
        draw.rectangle(
            (header_x1, header_y1, header_x2, header_y2),
            fill=(255, 255, 255),
            outline=border_light,
            width=3,
        )

    draw.text((margin + 24, 70), "Javoblar varaqi", font=title_font, fill=text_dark)
    draw.line((margin + 24, 140, width - margin - 24, 140), fill=border_light, width=3)

    # Small warning/instructions box (light color; far from detection ROIs)
    warn_x1 = margin + 24
    warn_y1 = 160
    warn_x2 = width - margin - 24
    warn_y2 = 220
    try:
        draw.rounded_rectangle(
            (warn_x1, warn_y1, warn_x2, warn_y2),
            radius=12,
            fill=warning_bg,
            outline=warning_border,
            width=2,
        )
    except Exception:
        draw.rectangle(
            (warn_x1, warn_y1, warn_x2, warn_y2),
            fill=warning_bg,
            outline=warning_border,
            width=2,
        )
    draw.text(
        (warn_x1 + 18, warn_y1 + 14),
        "Ko'rsatma: Har bir savolda faqat 1 ta variantni bo'yang. Test ID ni pastdagi kataklarda belgilang.",
        font=tiny_font,
        fill=text_dark,
    )

    # Personal info (simple, scanner-safe): single line for Name/Surname
    name_y = 248
    name_x1 = margin + 24
    name_x2 = width - margin - 24
    draw.text((name_x1, name_y), "Ism, Familiya:", font=label_font, fill=text_dark)
    # writing line starts after the label
    line_start_x = name_x1 + 260
    line_y = name_y + 44
    draw.line((line_start_x, line_y, name_x2, line_y), fill=border_light, width=3)

    # Accent separator above answer grid
    draw.line((margin, start_y - 16, width - margin, start_y - 16), fill=accent_blue, width=4)

    # Questions layout (4 columns fits A4, leaves space for code grid)
    num_columns = 4
    rows_per_col = int(math.ceil(num_questions / num_columns))
    col_width = (width - 2 * margin) // num_columns

    bubble_size = 54
    gap_x = 22
    gap_y = 18
    qnum_offset_x = 0
    bubbles_offset_x = 90
    outline_width = 4

    variant_labels = [chr(65 + i) for i in range(num_variants)]

    # Light per-column backgrounds (very light so thresholding won't treat as marks)
    col_bg_colors = [
        (240, 248, 255),  # aliceblue
        (248, 249, 250),  # #f8f9fa
        (245, 255, 250),  # mintcream
        (255, 250, 240),  # floralwhite
    ]
    answers_top = start_y - 8
    answers_bottom = start_y + rows_per_col * (bubble_size + gap_y) + 8
    for col in range(num_columns):
        x1 = start_x + col * col_width
        x2 = x1 + col_width - 8
        draw.rectangle((x1, answers_top, x2, answers_bottom), fill=col_bg_colors[col % len(col_bg_colors)])

    for col in range(num_columns):
        x_base = start_x + col * col_width
        for row in range(rows_per_col):
            q_index = col * rows_per_col + row
            if q_index >= num_questions:
                continue
            y = start_y + row * (bubble_size + gap_y)
            draw.text((x_base + qnum_offset_x, y + 8), f"{q_index + 1:03}", font=label_font, fill="black")
            for v in range(num_variants):
                x = x_base + bubbles_offset_x + v * (bubble_size + gap_x)
                draw.ellipse(
                    (x, y, x + bubble_size, y + bubble_size),
                    outline="black",
                    width=outline_width,
                )
                draw.text(
                    (x + bubble_size // 2 - 12, y + bubble_size // 2 - 18),
                    variant_labels[v],
                    font=small_font,
                    fill="black",
                )

    # Test ID grid (code_digits x 10)
    grid_left_x = margin
    grid_top_y = height - 760
    # Make Test ID bubbles same size as answer bubbles
    grid_cell_size = bubble_size
    grid_gap = 12
    draw.text(
        (grid_left_x, grid_top_y - 64),
        f"Test ID ({code_digits} xonali)",
        font=label_font,
        fill="black",
    )

    # Column labels 0-9
    for digit in range(10):
        x = grid_left_x + digit * (grid_cell_size + grid_gap)
        draw.text(
            (x + grid_cell_size // 2 - 8, grid_top_y - 30),
            str(digit),
            font=small_font,
            fill="black",
        )

    for row in range(code_digits):
        y = grid_top_y + row * (grid_cell_size + grid_gap)
        # Row label (position index)
        draw.text(
            (grid_left_x - 40, y + grid_cell_size // 2 - 14),
            str(row + 1),
            font=small_font,
            fill="black",
        )
        for digit in range(10):
            x = grid_left_x + digit * (grid_cell_size + grid_gap)
            draw.ellipse(
                (x, y, x + grid_cell_size, y + grid_cell_size),
                outline="black",
                width=3,
            )

    os.makedirs(os.path.dirname(sheet_path) or ".", exist_ok=True)
    img.save(sheet_path)
    print(f"OMR varaqi saqlandi: {sheet_path}")

if __name__ == "__main__":
    generate_omr_sheet("omr_example_108.png", num_questions=108, code_digits=5)
