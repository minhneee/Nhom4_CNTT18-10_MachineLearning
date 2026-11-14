"""
Kiểm thử bước tiền xử lý ảnh của GUI (gui_demo.py) trên ảnh NGOÀI bộ dữ liệu:
  1. Ảnh vẽ trên canvas (nền trắng, nét đen) - mô phỏng bằng font chữ với 2 độ dày nét.
  2. Ảnh chụp chữ viết trên giấy - mô phỏng nền giấy xám, ánh sáng không đều, nhiễu,
     mờ nhẹ, giấy kẻ dòng, bóng đổ.
Mỗi trường hợp đo: độ chính xác 1 chữ số (0-9) và độ chính xác cả chuỗi nhiều chữ số.
Kết quả in ra màn hình và ghi vào report/gui_eval.tex để đưa vào báo cáo.

ponytail: dùng font hệ thống thay cho chữ viết tay thật - chỉ là ước lượng tương đối,
          không thay thế được việc thử với ảnh viết tay thật của người dùng.
"""
import os
import sys
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from gui_demo import preprocess_digits, STROKE_RADIUS, CANVAS_STROKE_RADIUS  # noqa: E402

model = joblib.load(os.path.join(BASE_DIR, 'best_model.pkl'))
rng = np.random.default_rng(42)
FONTS = []   # chỉ dùng các font có trên máy
for f in ["comic.ttf", "segoepr.ttf", "segoesc.ttf", "arial.ttf", "times.ttf", "calibri.ttf",
          "DejaVuSans.ttf", "DejaVuSerif.ttf"]:
    try:
        ImageFont.truetype(f, 10)
        FONTS.append(f)
    except OSError:
        pass
STROKES = (0, 3)
MULTI = ["23", "2026", "907", "1458", "36"]


def canvas(text, font, stroke, W=440, H=200):
    """Ảnh giống canvas GUI: nền trắng, nét đen, các chữ số viết cách nhau."""
    img = Image.new('L', (W, H), 255)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(font, 150 if len(text) == 1 else 120)
    step = W // (len(text) + 1)
    for i, ch in enumerate(text):
        d.text((step * (i + 1), H // 2), ch, fill=0, font=f, anchor='mm', stroke_width=stroke + 2, stroke_fill=0)
    return img


def photo(text, font, stroke, ruled=False, shadow=False, W=1600, H=1200):
    """Ảnh chụp giấy: nền xám không đều + nhiễu + mờ, tùy chọn dòng kẻ và bóng đổ."""
    yy, xx = np.mgrid[0:H, 0:W]
    paper = 150 + 42 * (xx / W) + 28 * (yy / H)
    if shadow:
        paper -= 70 * np.exp(-(((xx - W) / 500.0) ** 2 + (yy / 400.0) ** 2))
    ink = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(ink)
    if ruled:
        for y in range(80, H, 110):
            d.line([(0, y), (W, y)], fill=120, width=3)
    f = ImageFont.truetype(font, 380 if len(text) == 1 else 300)
    step = W // (len(text) + 1)
    for i, ch in enumerate(text):
        d.text((step * (i + 1), H // 2), ch, fill=255, font=f, anchor='mm', stroke_width=stroke, stroke_fill=255)
    a = np.array(ink.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255
    img = paper * (1 - 0.65 * a) + rng.normal(0, 6, (H, W))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert('RGB')


def predict_text(img, stroke_radius):
    return ''.join(str(model.predict(f)[0]) for f, _ in preprocess_digits(img, stroke_radius=stroke_radius))


def evaluate(make, stroke_radius):
    ok1 = n1 = okm = nm = 0
    for font in FONTS:
        for s in STROKES:
            for dg in range(10):
                ok1 += predict_text(make(str(dg), font, s), stroke_radius) == str(dg)
                n1 += 1
            for t in MULTI:
                okm += predict_text(make(t, font, s), stroke_radius) == t
                nm += 1
    return ok1, n1, okm, nm


# Dùng đúng chế độ như GUI: canvas giữ nguyên nét, ảnh tải lên chuẩn hóa độ dày nét
cases = [
    ("Vẽ trên canvas", canvas, CANVAS_STROKE_RADIUS),
    ("Ảnh chụp, giấy trơn", photo, STROKE_RADIUS),
    ("Ảnh chụp, giấy kẻ dòng", lambda t, f, s: photo(t, f, s, ruled=True), STROKE_RADIUS),
    ("Ảnh chụp, có bóng đổ", lambda t, f, s: photo(t, f, s, shadow=True), STROKE_RADIUS),
]
pct = lambda a, b: f"{a / b * 100:.1f}".replace('.', ',')
rows = []
for name, make, sr in cases:
    ok1, n1, okm, nm = evaluate(make, sr)
    print(f"{name:26s} 1 chữ số: {ok1}/{n1} ({ok1/n1:.1%}) | chuỗi nhiều chữ số: {okm}/{nm} ({okm/nm:.1%})")
    rows.append(f"{name} & {ok1}/{n1} & {pct(ok1, n1)} & {okm}/{nm} & {pct(okm, nm)} \\\\\n")

# Chỉ ghi các dòng số liệu; khung bảng nằm trong report.tex để thống nhất định dạng
os.makedirs(os.path.join(BASE_DIR, 'report'), exist_ok=True)
with open(os.path.join(BASE_DIR, 'report', 'gui_eval.tex'), 'w', encoding='utf-8') as f:
    f.writelines(rows)
print(f"Đã ghi report/gui_eval.tex ({len(FONTS)} font, {len(STROKES)} độ dày nét, chuỗi: {', '.join(MULTI)})")
