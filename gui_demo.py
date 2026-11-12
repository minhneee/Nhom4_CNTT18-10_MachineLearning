"""
Nhận dạng chữ số viết tay - Demo GUI
=====================================
Ứng dụng demo sử dụng Tkinter cho phép người dùng:
- Vẽ chữ số trên canvas
- Tải ảnh từ file
- Nhận dạng chữ số bằng mô hình đã huấn luyện
- Hiển thị kết quả dự đoán với biểu đồ độ tin cậy
- Xem mẫu ngẫu nhiên từ tập test cùng dự đoán

Tác giả: Nhóm sinh viên
Môn: Học máy
"""

import os
import sys

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageFilter, ImageOps
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import joblib
import warnings
warnings.filterwarnings('ignore')

# Đường dẫn thư mục gốc của project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model():
    """
    Tải mô hình đã huấn luyện trực tiếp từ best_model.pkl.
    Trả về: (model, None, model_info_dict)
    """
    model_path = os.path.join(BASE_DIR, 'best_model.pkl')
    info = {'source': '', 'accuracy': None}
    model = None

    if os.path.exists(model_path):
        print("[INFO] Đang tải mô hình từ best_model.pkl...")
        model = joblib.load(model_path)
        info['source'] = 'best_model.pkl (Huấn luyện từ dataset_images)'

        # Đánh giá độ chính xác thực tế trên tập dữ liệu ảnh
        try:
            data_dir = os.path.join(BASE_DIR, 'dataset_images')
            if os.path.exists(data_dir):
                test_imgs, test_labels = [], []
                for c in sorted(os.listdir(data_dir)):
                    cdir = os.path.join(data_dir, c)
                    if not os.path.isdir(cdir): continue
                    for f in sorted(os.listdir(cdir)):
                        if f.endswith('.png'):
                            im = Image.open(os.path.join(cdir, f)).convert('L')
                            test_imgs.append(np.array(im, dtype=np.float32).flatten() / 255.0)
                            test_labels.append(int(c))
                if len(test_imgs) > 0:
                    from sklearn.model_selection import train_test_split
                    _, X_te, _, y_te = train_test_split(
                        np.array(test_imgs), np.array(test_labels),
                        test_size=0.2, random_state=42, stratify=test_labels
                    )
                    info['accuracy'] = model.score(X_te, y_te)
        except Exception:
            pass

        print("[INFO] Tải mô hình thành công!")
    else:
        print("[INFO] Không tìm thấy best_model.pkl. Đang huấn luyện nhanh mô hình KNN...")
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import train_test_split

        data_dir = os.path.join(BASE_DIR, 'dataset_images')
        images, labels = [], []
        for c in sorted(os.listdir(data_dir)):
            cdir = os.path.join(data_dir, c)
            if not os.path.isdir(cdir): continue
            for f in sorted(os.listdir(cdir)):
                if f.endswith('.png'):
                    im = Image.open(os.path.join(cdir, f)).convert('L')
                    images.append(np.array(im, dtype=np.float32).flatten() / 255.0)
                    labels.append(int(c))

        X_tr, X_te, y_tr, y_te = train_test_split(
            np.array(images), np.array(labels), test_size=0.2, random_state=42, stratify=labels
        )
        model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        model.fit(X_tr, y_tr)
        info['source'] = 'K-NN (Huấn luyện tự động từ dataset_images)'
        info['accuracy'] = model.score(X_te, y_te)
        joblib.dump(model, model_path)
        print(f"[INFO] Đã huấn luyện xong mô hình KNN (Test Acc: {info['accuracy']:.4f})")

    return model, None, info


def preprocess_image(image):
    """
    Tiền xử lý ảnh đầu vào đồng bộ chuẩn với bộ dữ liệu ảnh:
    1. Chuyển sang ảnh xám (Grayscale).
    2. Nếu ảnh gốc đã là 8x8 (từ dataset_images): chuẩn hóa pixel [0, 1].
    3. Nếu là ảnh vẽ canvas hoặc ảnh kích thước lớn:
       - Đảo ngược màu nếu nền trắng (nền đen 0, nét vẽ sáng 255).
       - Chuẩn hóa giống bộ optdigits: xem _digit_to_features.
    Trả về: (features_1x64, arr_8x8)
    """
    if image.mode != 'L':
        image = image.convert('L')

    # Trường hợp ảnh chuẩn 8x8 (từ bộ dataset_images)
    if image.size == (8, 8):
        arr = np.array(image, dtype=np.float32)
        if arr.mean() > 128:
            arr = 255.0 - arr
        arr = arr / 255.0
        return arr.flatten().reshape(1, -1), arr

    # Trường hợp ảnh vẽ canvas hoặc ảnh kích thước bất kỳ: coi toàn bộ nét là 1 chữ số
    ink = _ink_array(image)
    if ink.max() <= INK_THRESHOLD:
        return np.zeros((1, 64), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)
    return _digit_to_features(ink)


INK_THRESHOLD = 50   # pixel (0-255, nét sáng) lớn hơn ngưỡng này được coi là nét vẽ
STROKE_RADIUS = 3    # bán kính nét sau chuẩn hóa (khung 32x32) cho ẢNH TẢI LÊN / ẢNH CHỤP
CANVAS_STROKE_RADIUS = 0  # ảnh vẽ chuột: nét đã đều -> giữ nguyên (thực nghiệm: 82% so với 77% khi chuẩn hóa)


def _ink_array(image, max_side=800):
    """
    Ảnh PIL (canvas hoặc ẢNH CHỤP giấy) -> mảng nhị phân 0/255, nét = 255.
    Ảnh chụp có nền giấy xám, ánh sáng không đều, nhiễu, dòng kẻ nên không thể
    dùng ngưỡng cố định:
      1. Thu nhỏ (cạnh dài <= max_side) để xử lý nhanh và ổn định.
      2. Trừ nền: ước lượng độ sáng nền bằng phép lọc trung vị kích thước lớn
         -> khử ánh sáng không đều / bóng đổ.
      3. Ngưỡng Otsu tự động trên ảnh đã trừ nền.
      4. Xóa dòng kẻ ngang dài (giấy kẻ ô/kẻ dòng) và các đốm nhiễu nhỏ.
    """
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    from skimage.morphology import remove_small_objects, binary_opening, footprint_rectangle

    gray = image.convert('L')
    scale = min(1.0, max_side / max(gray.size))
    if scale < 1.0:
        gray = gray.resize((round(gray.width * scale), round(gray.height * scale)), Image.Resampling.LANCZOS)
    g = np.array(gray, dtype=np.float32)

    # Nền tối, nét sáng (vd. ảnh từ dataset) -> đảo về nền sáng, nét tối
    if np.median(g) < 128:
        g = 255.0 - g

    # Trừ nền: ước lượng nền ở độ phân giải 1/8 (MaxFilter xóa nét tối, MedianFilter làm mượt)
    h, w = g.shape
    small = Image.fromarray(g.astype(np.uint8)).resize((max(1, w // 8), max(1, h // 8)), Image.Resampling.BOX)
    k = max(5, min(small.size) // 3) | 1      # cửa sổ ~1/3 cạnh ảnh: lớn hơn nét dày nhất của chữ số
    small = small.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MedianFilter(5))
    background = np.array(small.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
    diff = np.clip(background - g, 0, None)
    if diff.max() < 20:                       # không có nét nào đủ tương phản
        return np.zeros_like(g)
    binary = diff > max(threshold_otsu(diff), 20)

    # Xóa dòng kẻ giấy: đường ngang DÀI (>= 60% chiều rộng ảnh) và MẢNH (dày < t pixel).
    # Nét ngang của chữ số (số 2, 4, 5, 7) ngắn hơn hoặc dày hơn nên được giữ lại.
    lines = binary_opening(binary, footprint_rectangle((1, max(30, int(0.6 * g.shape[1])))))
    t = max(3, g.shape[0] // 100)
    lines &= ~binary_opening(lines, footprint_rectangle((t, 1)))
    binary &= ~lines

    # Xóa đốm nhiễu: thành phần nhỏ hơn 1% thành phần lớn nhất
    lab = label(binary)
    if lab.max() > 0:
        largest = np.bincount(lab.ravel())[1:].max()
        binary = remove_small_objects(lab, min_size=max(10, largest // 100)) > 0
    return binary.astype(np.float32) * 255.0


def _digit_to_features(ink, stroke_radius=STROKE_RADIUS):
    """
    Chuẩn hóa 1 chữ số theo đúng cách tạo bộ UCI optdigits (load_digits):
    - Cắt sát khung nét vẽ, phóng cho chữ số CHIẾM HẾT CHIỀU CAO khung 32x32,
      căn giữa theo chiều ngang (dataset: cao 8/8 pixel, rộng ~6/8 pixel).
    - Thu nhỏ 32x32 -> 8x8 bằng trung bình khối 4x4 (Image.BOX), giống phép
      đếm pixel trong từng khối 4x4 của bộ dữ liệu gốc.
    - Chia cho giá trị lớn nhất (giống ảnh PNG trong dataset_images).
    Trả về: (features_1x64 trong [0, 1], arr_8x8 trong [0, 1])
    """
    rows = np.where(ink.max(axis=1) > INK_THRESHOLD)[0]
    cols = np.where(ink.max(axis=0) > INK_THRESHOLD)[0]
    from skimage.morphology import skeletonize, binary_dilation, disk

    crop = (ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] > INK_THRESHOLD).astype(np.uint8) * 255
    h, w = crop.shape
    scale = min(32 / h, 32 / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    digit = np.array(Image.fromarray(crop).resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float32)

    frame = np.zeros((32, 32), dtype=np.float32)
    frame[(32 - nh) // 2:(32 - nh) // 2 + nh, (32 - nw) // 2:(32 - nw) // 2 + nw] = digit
    # Chuẩn hóa độ dày nét: nét bút bi mảnh hay nét chuột dày đều được đưa về
    # bộ xương 1 pixel rồi làm dày lại với bán kính stroke_radius (giống nét trong optdigits)
    if stroke_radius:
        frame = binary_dilation(skeletonize(frame > 64), disk(stroke_radius)).astype(np.float32) * 255
    frame = Image.fromarray(frame.astype(np.uint8))
    arr_8x8 = np.array(frame.resize((8, 8), Image.Resampling.BOX), dtype=np.float32)
    arr_8x8 = arr_8x8 / arr_8x8.max() if arr_8x8.max() > 0 else arr_8x8
    return arr_8x8.reshape(1, -1), arr_8x8


def preprocess_digits(image, min_gap=4, min_ink=30, stroke_radius=STROKE_RADIUS):
    """
    Tách ảnh chứa NHIỀU chữ số viết liền hàng (vd: "2026") thành từng chữ số
    dựa trên hình chiếu theo cột: các cột trống liên tiếp >= min_gap pixel là
    khoảng cách giữa 2 chữ số. Mỗi chữ số được chuẩn hóa bằng _digit_to_features.
    Trả về: list các (features_1x64, arr_8x8) theo thứ tự từ trái sang phải.
    """
    if image.size == (8, 8):
        return [preprocess_image(image)]

    ink = _ink_array(image)
    has_ink = (ink > INK_THRESHOLD).any(axis=0)

    # Tìm các đoạn cột có nét, nối các đoạn cách nhau < min_gap cột trống
    segments, start, last = [], None, None
    for c in np.where(has_ink)[0]:
        if start is None:
            start = c
        elif c - last > min_gap:
            segments.append((start, last))
            start = c
        last = c
    if start is not None:
        segments.append((start, last))

    digits = []
    for c0, c1 in segments:
        part = ink[:, c0:c1 + 1]
        if (part > INK_THRESHOLD).sum() < min_ink:   # bỏ qua chấm nhiễu nhỏ
            continue
        # Các chữ số viết dính nhau: đoạn rộng bất thường so với chiều cao
        # (1 chữ số thường rộng khoảng DIGIT_ASPECT lần chiều cao) -> tách thành n phần
        rows = np.where((part > INK_THRESHOLD).any(axis=1))[0]
        h, w = rows[-1] - rows[0] + 1, part.shape[1]
        n = max(1, round(w / (DIGIT_ASPECT * h)))
        for piece in _split_touching(part, n):
            if (piece > INK_THRESHOLD).sum() >= min_ink:
                digits.append(_digit_to_features(piece, stroke_radius))
    return digits


DIGIT_ASPECT = 0.85   # tỉ lệ rộng/cao ước lượng của một chữ số viết tay (chọn bằng thực nghiệm)


def _split_touching(part, n):
    """
    Tách đoạn ảnh chứa n chữ số viết dính nhau. Thay vì cắt đều, mỗi vết cắt được
    đặt tại cột có ÍT MỰC NHẤT (khe giữa hai chữ số) trong lân cận vị trí cắt đều.
    """
    if n <= 1:
        return [part]
    profile = (part > INK_THRESHOLD).sum(axis=0)
    W = part.shape[1]
    radius = max(1, int(W / n * 0.3))
    cuts = [0]
    for k in range(1, n):
        center = int(k * W / n)
        lo, hi = max(cuts[-1] + 1, center - radius), min(W - 1, center + radius)
        cuts.append(lo + int(np.argmin(profile[lo:hi + 1])))
    cuts.append(W)
    return [part[:, a:b] for a, b in zip(cuts[:-1], cuts[1:])]


class DigitRecognizerApp:
    """Ứng dụng nhận dạng chữ số viết tay"""

    def __init__(self, root):
        self.root = root
        self.root.title("Nhận dạng chữ số viết tay - Demo")
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(True, True)

        # Tải mô hình
        self.model, self.scaler, self.model_info = load_model()

        # Canvas drawing state
        self.canvas_size = 200   # chiều cao canvas
        self.canvas_w = 440      # canvas rộng để viết được nhiều chữ số
        self.brush_size = 18
        self.drawing = False
        self.last_x = None
        self.last_y = None

        # Tạo ảnh PIL để vẽ (nền trắng)
        self.draw_image = Image.new('L', (self.canvas_w, self.canvas_size), 255)
        self.draw_ctx = ImageDraw.Draw(self.draw_image)

        # Thiết lập giao diện
        self._setup_styles()
        self._create_menu()
        self._create_widgets()
        self._create_status_bar()
        self._load_sample_images()

        # Cập nhật kích thước cửa sổ
        self.root.update_idletasks()
        self.root.minsize(950, 700)

    def _setup_styles(self):
        """Thiết lập style cho giao diện"""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.colors = {
            'primary': '#2196F3',
            'primary_dark': '#1976D2',
            'accent': '#FF5722',
            'bg': '#f5f5f5',
            'card': '#ffffff',
            'text': '#333333',
            'text_light': '#666666',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336',
            'border': '#e0e0e0',
            'status_bg': '#263238',
            'status_fg': '#B0BEC5',
        }

        self.style.configure('Title.TLabel',
                             font=('Segoe UI', 16, 'bold'),
                             foreground=self.colors['primary_dark'],
                             background=self.colors['bg'])
        self.style.configure('Header.TLabel',
                             font=('Segoe UI', 11, 'bold'),
                             foreground=self.colors['text'],
                             background=self.colors['bg'])
        self.style.configure('Info.TLabel',
                             font=('Segoe UI', 10),
                             foreground=self.colors['text_light'],
                             background=self.colors['bg'])
        self.style.configure('Card.TFrame', background=self.colors['card'])
        self.style.configure('TLabelframe', background=self.colors['bg'])
        self.style.configure('TLabelframe.Label',
                             font=('Segoe UI', 10, 'bold'),
                             foreground=self.colors['primary_dark'],
                             background=self.colors['bg'])

    def _create_menu(self):
        """Tạo thanh menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Tải ảnh...", command=self._load_image, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.root.quit, accelerator="Alt+F4")
        menubar.add_cascade(label="Tệp", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Xóa canvas", command=self._clear_canvas, accelerator="Ctrl+D")
        edit_menu.add_command(label="Mẫu ngẫu nhiên", command=self._show_random_test_samples)
        menubar.add_cascade(label="Chỉnh sửa", menu=edit_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Hướng dẫn sử dụng", command=self._show_help)
        help_menu.add_separator()
        help_menu.add_command(label="Về chương trình...", command=self._show_about)
        menubar.add_cascade(label="Trợ giúp", menu=help_menu)

        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._load_image())
        self.root.bind('<Control-d>', lambda e: self._clear_canvas())

    def _create_status_bar(self):
        """Tạo thanh trạng thái ở dưới cùng"""
        status_frame = tk.Frame(self.root, bg=self.colors['status_bg'], height=28)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        status_frame.pack_propagate(False)

        # Model info
        model_name = type(self.model).__name__ if self.model else "Chưa tải"
        model_detail = model_name
        if hasattr(self.model, 'n_neighbors'):
            model_detail += f" (k={self.model.n_neighbors})"
        elif hasattr(self.model, 'max_depth'):
            depth = self.model.max_depth if self.model.max_depth else "None"
            model_detail += f" (max_depth={depth})"

        acc_text = ""
        if self.model_info.get('accuracy') is not None:
            acc_text = f"  |  Độ chính xác: {self.model_info['accuracy']:.2%}"

        source_text = ""
        if self.model_info.get('source'):
            source_text = f"  |  Nguồn: {self.model_info['source']}"

        self.status_label = tk.Label(
            status_frame,
            text=f"  🤖 Mô hình: {model_detail}{acc_text}{source_text}",
            font=('Segoe UI', 9),
            fg=self.colors['status_fg'],
            bg=self.colors['status_bg'],
            anchor='w'
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_right = tk.Label(
            status_frame,
            text="Đề tài 5 - Học máy  ",
            font=('Segoe UI', 9),
            fg='#78909C',
            bg=self.colors['status_bg'],
            anchor='e'
        )
        self.status_right.pack(side=tk.RIGHT)

    def _create_widgets(self):
        """Tạo tất cả widget cho giao diện"""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.configure(style='Card.TFrame')

        # Tiêu đề
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(title_frame,
                               text="🔢 Nhận dạng chữ số viết tay",
                               font=('Segoe UI', 18, 'bold'),
                               fg=self.colors['primary_dark'],
                               bg=self.colors['card'])
        title_label.pack(side=tk.LEFT)

        subtitle_label = tk.Label(title_frame,
                                  text="Vẽ 1 hoặc nhiều chữ số (viết cách nhau) hoặc tải ảnh để nhận dạng",
                                  font=('Segoe UI', 10),
                                  fg=self.colors['text_light'],
                                  bg=self.colors['card'])
        subtitle_label.pack(side=tk.LEFT, padx=(15, 0), pady=(8, 0))

        # Content area - 3 columns
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # ===== CỘT 1: Vẽ chữ số =====
        left_frame = ttk.LabelFrame(content_frame, text="  ✏️ Vẽ chữ số  ", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        # Canvas vẽ
        canvas_container = ttk.Frame(left_frame)
        canvas_container.pack(pady=(0, 8))

        self.canvas = tk.Canvas(
            canvas_container,
            width=self.canvas_w,
            height=self.canvas_size,
            bg='white',
            cursor='crosshair',
            highlightthickness=2,
            highlightbackground=self.colors['primary'],
            relief='flat'
        )
        self.canvas.pack()

        # Gắn sự kiện chuột
        self.canvas.bind('<Button-1>', self._start_draw)
        self.canvas.bind('<B1-Motion>', self._draw)
        self.canvas.bind('<ButtonRelease-1>', self._stop_draw)

        # Slider kích thước bút vẽ
        brush_frame = ttk.Frame(left_frame)
        brush_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(brush_frame, text="Kích thước bút:",
                 font=('Segoe UI', 9), bg=self.colors['bg']).pack(side=tk.LEFT)

        self.brush_slider = tk.Scale(
            brush_frame, from_=8, to=32,
            orient=tk.HORIZONTAL, length=130,
            command=self._update_brush_size,
            bg=self.colors['bg'],
            highlightthickness=0,
            troughcolor=self.colors['border']
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT, padx=(5, 0))

        # Nút chức năng
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        predict_btn = tk.Button(
            btn_frame, text="🔍 Nhận dạng",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['primary'], fg='white',
            activebackground=self.colors['primary_dark'],
            activeforeground='white',
            relief='flat', padx=12, pady=6,
            cursor='hand2',
            command=self._predict_drawn
        )
        predict_btn.pack(fill=tk.X, pady=(0, 4))

        clear_btn = tk.Button(
            btn_frame, text="🗑️ Xóa canvas",
            font=('Segoe UI', 10),
            bg='#e0e0e0', fg=self.colors['text'],
            activebackground='#bdbdbd',
            relief='flat', padx=12, pady=6,
            cursor='hand2',
            command=self._clear_canvas
        )
        clear_btn.pack(fill=tk.X, pady=(0, 4))

        load_btn = tk.Button(
            btn_frame, text="📂 Tải ảnh từ file",
            font=('Segoe UI', 10),
            bg='#e8f5e9', fg='#2E7D32',
            activebackground='#c8e6c9',
            relief='flat', padx=12, pady=6,
            cursor='hand2',
            command=self._load_image
        )
        load_btn.pack(fill=tk.X, pady=(0, 4))

        random_btn = tk.Button(
            btn_frame, text="🎲 Mẫu ngẫu nhiên",
            font=('Segoe UI', 10),
            bg='#FFF3E0', fg='#E65100',
            activebackground='#FFE0B2',
            relief='flat', padx=12, pady=6,
            cursor='hand2',
            command=self._show_random_test_samples
        )
        random_btn.pack(fill=tk.X)

        # ===== CỘT 2: Kết quả =====
        mid_frame = ttk.LabelFrame(content_frame, text="  📊 Kết quả nhận dạng  ", padding=10)
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Kết quả dự đoán
        result_container = ttk.Frame(mid_frame)
        result_container.pack(fill=tk.X, pady=(0, 5))

        tk.Label(result_container, text="Chữ số dự đoán:",
                 font=('Segoe UI', 11), bg=self.colors['bg']).pack(side=tk.LEFT)

        self.result_label = tk.Label(
            result_container, text="—",
            font=('Segoe UI', 32, 'bold'),
            fg=self.colors['accent'],
            bg=self.colors['bg']
        )
        self.result_label.pack(side=tk.LEFT, padx=(10, 0))

        self.confidence_label = tk.Label(
            result_container, text="",
            font=('Segoe UI', 11, 'bold'),
            fg=self.colors['success'],
            bg=self.colors['bg']
        )
        self.confidence_label.pack(side=tk.LEFT, padx=(10, 0))

        # Biểu đồ độ tin cậy — HORIZONTAL bar chart
        self.fig = Figure(figsize=(4.5, 3.5), dpi=90, facecolor=self.colors['bg'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['bg'])
        self._init_chart()

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=mid_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Ảnh đã xử lý (preview 8x8)
        preview_frame = ttk.Frame(mid_frame)
        preview_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(preview_frame, text="Ảnh sau xử lý (8×8):",
                 font=('Segoe UI', 9), bg=self.colors['bg']).pack(side=tk.LEFT)

        self.preview_label = tk.Label(preview_frame, bg=self.colors['bg'])
        self.preview_label.pack(side=tk.LEFT, padx=(5, 0))

        # ===== CỘT 3: Mẫu từ dataset =====
        right_frame = ttk.LabelFrame(content_frame, text="  🖼️ Mẫu từ bộ dữ liệu  ", padding=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))

        self.sample_frame = ttk.Frame(right_frame)
        self.sample_frame.pack(fill=tk.BOTH, expand=True)

    def _init_chart(self):
        """Khởi tạo biểu đồ ngang mặc định"""
        self.ax.clear()
        digits = list(range(10))
        labels = [str(d) for d in digits]
        self.ax.barh(labels, [0] * 10, color='#bbdefb', edgecolor='#90caf9', height=0.7)
        self.ax.set_xlabel('Độ tin cậy (%)', fontsize=9)
        self.ax.set_ylabel('Chữ số', fontsize=9)
        self.ax.set_title('Độ tin cậy cho từng chữ số', fontsize=10, fontweight='bold')
        self.ax.set_xlim(0, 105)
        self.ax.invert_yaxis()
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.fig.tight_layout()

    def _get_confidence_color(self, value):
        """Trả về màu dựa trên độ tin cậy: xanh lá (cao) → vàng (trung bình) → đỏ (thấp)"""
        if value >= 60:
            return self.colors['success']   # Xanh lá
        elif value >= 25:
            return self.colors['warning']   # Vàng cam
        else:
            return '#e0e0e0'                # Xám nhạt

    def _update_chart(self, probabilities):
        """Cập nhật biểu đồ ngang với xác suất dự đoán, sắp xếp theo độ tin cậy"""
        self.ax.clear()
        digits = list(range(10))
        probs_pct = probabilities * 100

        # Màu: xanh lá cho cao, vàng cho trung bình, xám cho thấp, xanh đậm cho top
        best_idx = np.argmax(probs_pct)
        colors = []
        for i, val in enumerate(probs_pct):
            if i == best_idx:
                colors.append(self.colors['primary'])
            else:
                colors.append(self._get_confidence_color(val))

        labels = [str(d) for d in digits]
        bars = self.ax.barh(labels, probs_pct, color=colors,
                            edgecolor='#90caf9', height=0.7)

        # Hiển thị giá trị bên phải mỗi thanh
        for bar, val in zip(bars, probs_pct):
            if val > 1:
                self.ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                             f'{val:.1f}%', ha='left', va='center',
                             fontsize=7, fontweight='bold')

        self.ax.set_xlabel('Độ tin cậy (%)', fontsize=9)
        self.ax.set_ylabel('Chữ số', fontsize=9)
        self.ax.set_title('Độ tin cậy cho từng chữ số', fontsize=10, fontweight='bold')
        self.ax.set_xlim(0, 115)
        self.ax.invert_yaxis()
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.fig.tight_layout()
        self.chart_canvas.draw()

    def _load_sample_images(self):
        """Tải và hiển thị mẫu từ bộ dữ liệu dataset_images, có thể bấm để nhận dạng ngay"""
        sample_dir = os.path.join(BASE_DIR, 'dataset_images')

        for row in range(5):
            for col in range(2):
                digit = row * 2 + col
                if digit > 9:
                    break

                frame = ttk.Frame(self.sample_frame)
                frame.grid(row=row, column=col, padx=2, pady=2, sticky='nsew')

                digit_dir = os.path.join(sample_dir, str(digit))
                img_to_use = None
                if os.path.isdir(digit_dir):
                    files = sorted([f for f in os.listdir(digit_dir) if f.endswith('.png')])
                    if files:
                        try:
                            img_to_use = Image.open(os.path.join(digit_dir, files[0]))
                        except Exception:
                            pass

                if img_to_use is None:
                    try:
                        from sklearn.datasets import load_digits
                        digits_data = load_digits()
                        idx = np.where(digits_data.target == digit)[0][0]
                        img_arr = (digits_data.images[idx] / 16.0 * 255).astype(np.uint8)
                        img_to_use = Image.fromarray(img_arr, mode='L')
                    except Exception:
                        pass

                if img_to_use is not None:
                    thumb = img_to_use.resize((26, 26), Image.NEAREST)
                    photo = ImageTk.PhotoImage(thumb)

                    btn = tk.Button(
                        frame, image=photo, text=f" Số {digit}", compound=tk.LEFT,
                        font=('Segoe UI', 8, 'bold'),
                        bg='#ffffff', fg=self.colors['text'],
                        activebackground='#e3f2fd',
                        relief='groove', padx=2, pady=1, cursor='hand2',
                        command=lambda d=digit, im=img_to_use: self._load_sample_digit(d, im)
                    )
                    btn.image = photo
                    btn.pack(fill=tk.X)

    def _load_sample_digit(self, digit, img):
        """Tải mẫu chữ số lên canvas và nhận dạng tức thì"""
        self._clear_canvas()
        display_img = img.convert('L').resize((self.canvas_size, self.canvas_size), Image.NEAREST)
        self._loaded_photo = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(
            self.canvas_w // 2, self.canvas_size // 2,
            image=self._loaded_photo
        )
        self.draw_image = img.copy()
        self._predict_from_image(img)

    # ========== Sự kiện vẽ ==========

    def _start_draw(self, event):
        """Bắt đầu vẽ khi nhấn chuột"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        # Vẽ điểm tại vị trí click
        r = self.brush_size // 2
        self.canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            fill='black', outline='black'
        )
        self.draw_ctx.ellipse(
            [event.x - r, event.y - r, event.x + r, event.y + r],
            fill=0
        )

    def _draw(self, event):
        """Vẽ khi kéo chuột — nội suy điểm cho nét mượt"""
        if self.drawing:
            x, y = event.x, event.y
            r = self.brush_size // 2

            # Nội suy giữa điểm cũ và điểm mới cho nét vẽ mượt
            if self.last_x is not None:
                # Vẽ đường trên canvas Tkinter
                self.canvas.create_line(
                    self.last_x, self.last_y, x, y,
                    fill='black', width=self.brush_size,
                    capstyle=tk.ROUND, joinstyle=tk.ROUND,
                    smooth=True
                )
                # Vẽ đường trên ảnh PIL
                self.draw_ctx.line(
                    [(self.last_x, self.last_y), (x, y)],
                    fill=0, width=self.brush_size
                )

            # Vẽ chấm tròn tại điểm hiện tại
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill='black', outline='black'
            )
            self.draw_ctx.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=0
            )

            self.last_x = x
            self.last_y = y

    def _stop_draw(self, event):
        """Dừng vẽ khi thả chuột"""
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def _update_brush_size(self, val):
        """Cập nhật kích thước bút vẽ"""
        self.brush_size = int(float(val))

    # ========== Chức năng chính ==========

    def _clear_canvas(self):
        """Xóa canvas và reset kết quả"""
        self.canvas.delete('all')
        self.draw_image = Image.new('L', (self.canvas_w, self.canvas_size), 255)
        self.draw_ctx = ImageDraw.Draw(self.draw_image)
        self.result_label.config(text="—")
        self.confidence_label.config(text="")
        self._init_chart()
        self.chart_canvas.draw()
        self.preview_label.config(image='')

    def _predict_drawn(self):
        """Nhận dạng chữ số từ canvas vẽ"""
        img_array = np.array(self.draw_image)
        if img_array.min() == 255:
            messagebox.showinfo("Thông báo", "Vui lòng vẽ chữ số trên canvas trước!")
            return
        self._predict_from_image(self.draw_image, stroke_radius=CANVAS_STROKE_RADIUS)

    def _load_image(self):
        """Tải ảnh từ file và nhận dạng"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh chữ số",
            filetypes=[
                ("Ảnh", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("Tất cả", "*.*")
            ],
            initialdir=os.path.join(BASE_DIR, 'dataset_images')
        )
        if not file_path:
            return

        try:
            # Ảnh chụp điện thoại lưu hướng xoay trong EXIF -> xoay lại cho đúng chiều
            self._show_and_predict_image(ImageOps.exif_transpose(Image.open(file_path)))
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở ảnh:\n{str(e)}")

    def _show_and_predict_image(self, image):
        """Hiển thị ảnh tải lên trên canvas (giữ nguyên tỉ lệ) và nhận dạng"""
        display_img = image.copy()
        display_img.thumbnail((self.canvas_w, self.canvas_size), Image.LANCZOS)
        self.canvas.delete('all')
        self._loaded_photo = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(self.canvas_w // 2, self.canvas_size // 2, image=self._loaded_photo)
        self.draw_image = image.copy()
        self._predict_from_image(image)

    def _predict_from_image(self, image, stroke_radius=STROKE_RADIUS):
        """Thực hiện dự đoán từ ảnh PIL"""
        try:
            digits = preprocess_digits(image, stroke_radius=stroke_radius)
            if not digits:
                messagebox.showinfo("Thông báo", "Không tìm thấy nét vẽ nào để nhận dạng!")
                return
            features = np.vstack([f for f, _ in digits])

            # Hiển thị các ảnh 8x8 đã xử lý cạnh nhau (phóng to để người dùng quan sát)
            gap = np.ones((8, 1), dtype=np.float32) * 0.5
            strip = np.hstack([part for _, a in digits for part in (a, gap)][:-1])
            disp_arr = np.clip(strip * 255.0, 0, 255).astype(np.uint8)
            preview_display = Image.fromarray(disp_arr, mode='L').resize((strip.shape[1] * 8, 64), Image.NEAREST)
            self._preview_photo = ImageTk.PhotoImage(preview_display)
            self.preview_label.config(image=self._preview_photo)

            # Dự đoán từng chữ số (trái -> phải)
            predictions = self.model.predict(features).astype(int)
            all_probs = self.model.predict_proba(features)

            # Hiển thị kết quả: chuỗi các chữ số; biểu đồ = xác suất của chữ số đầu tiên
            self.result_label.config(text=''.join(map(str, predictions)))
            prediction, probabilities = predictions[0], all_probs[0]
            confs = [p[d] * 100 for p, d in zip(all_probs, predictions)]
            max_conf = min(confs)   # màu theo chữ số kém tin cậy nhất

            # Mã màu cho độ tin cậy
            if max_conf >= 80:
                conf_color = self.colors['success']
            elif max_conf >= 50:
                conf_color = self.colors['warning']
            else:
                conf_color = self.colors['danger']

            conf_text = f"({confs[0]:.1f}%)" if len(confs) == 1 else \
                "(" + ", ".join(f"{d}: {c:.0f}%" for d, c in zip(predictions, confs)) + ")"
            self.confidence_label.config(text=conf_text, fg=conf_color)

            # Cập nhật biểu đồ
            self._update_chart(probabilities)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi nhận dạng:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _show_random_test_samples(self):
        """Hiển thị cửa sổ với mẫu ngẫu nhiên từ tập test dataset_images và dự đoán"""
        try:
            data_dir = os.path.join(BASE_DIR, 'dataset_images')
            test_imgs, test_labels, raw_pil_list = [], [], []

            for c in sorted(os.listdir(data_dir)):
                cdir = os.path.join(data_dir, c)
                if not os.path.isdir(cdir): continue
                for f in sorted(os.listdir(cdir)):
                    if f.endswith('.png'):
                        im = Image.open(os.path.join(cdir, f)).convert('L')
                        arr = np.array(im, dtype=np.float32) / 255.0
                        test_imgs.append(arr.flatten())
                        test_labels.append(int(c))
                        raw_pil_list.append(im)

            from sklearn.model_selection import train_test_split
            indices = list(range(len(test_imgs)))
            _, test_idx = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=test_labels
            )

            # Chọn 12 mẫu ngẫu nhiên từ test set
            n_samples = min(12, len(test_idx))
            sample_indices = random.sample(test_idx, n_samples)

            X_samples = np.array([test_imgs[i] for i in sample_indices])
            y_true = np.array([test_labels[i] for i in sample_indices])
            y_pred = self.model.predict(X_samples)

            # Tạo cửa sổ mới
            win = tk.Toplevel(self.root)
            win.title("Mẫu ngẫu nhiên từ tập test")
            win.configure(bg=self.colors['card'])
            win.resizable(False, False)

            tk.Label(win, text="🎲 Mẫu ngẫu nhiên từ tập kiểm tra",
                     font=('Segoe UI', 13, 'bold'),
                     fg=self.colors['primary_dark'],
                     bg=self.colors['card']).pack(pady=(10, 5))

            tk.Label(win, text="Nhãn thật (trên) — Dự đoán (dưới). Đỏ = sai, Xanh = đúng.",
                     font=('Segoe UI', 9),
                     fg=self.colors['text_light'],
                     bg=self.colors['card']).pack(pady=(0, 8))

            grid_frame = ttk.Frame(win)
            grid_frame.pack(padx=15, pady=(0, 15))

            self._sample_photos = []  # Keep references

            for i, idx in enumerate(range(n_samples)):
                row, col = divmod(i, 4)
                cell = ttk.Frame(grid_frame)
                cell.grid(row=row, column=col, padx=5, pady=5)

                # Ảnh
                img_array = (X_samples[idx].reshape(8, 8) * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
                img = img.resize((48, 48), Image.NEAREST)
                photo = ImageTk.PhotoImage(img)
                self._sample_photos.append(photo)

                tk.Label(cell, image=photo, bg=self.colors['bg'],
                         relief='solid', borderwidth=1).pack()

                correct = y_true[idx] == y_pred[idx]
                color = self.colors['success'] if correct else self.colors['danger']
                symbol = "✓" if correct else "✗"

                tk.Label(cell,
                         text=f"Thật: {y_true[idx]}  |  Đoán: {y_pred[idx]} {symbol}",
                         font=('Segoe UI', 8, 'bold'),
                         fg=color,
                         bg=self.colors['bg']).pack()

            # Accuracy summary
            correct_count = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            tk.Label(win,
                     text=f"Kết quả: {correct_count}/{n_samples} đúng ({correct_count/n_samples:.0%})",
                     font=('Segoe UI', 10, 'bold'),
                     fg=self.colors['primary_dark'],
                     bg=self.colors['card']).pack(pady=(0, 10))

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mẫu:\n{str(e)}")

    # ========== Dialogs ==========

    def _show_about(self):
        """Hiển thị hộp thoại Về chương trình"""
        about_win = tk.Toplevel(self.root)
        about_win.title("Về chương trình")
        about_win.configure(bg=self.colors['card'])
        about_win.resizable(False, False)
        about_win.geometry("420x350")

        # Căn giữa
        about_win.transient(self.root)
        about_win.grab_set()

        tk.Label(about_win, text="🔢",
                 font=('Segoe UI', 36),
                 bg=self.colors['card']).pack(pady=(20, 5))

        tk.Label(about_win,
                 text="Nhận dạng chữ số viết tay",
                 font=('Segoe UI', 14, 'bold'),
                 fg=self.colors['primary_dark'],
                 bg=self.colors['card']).pack()

        tk.Label(about_win,
                 text="Phiên bản 1.0",
                 font=('Segoe UI', 10),
                 fg=self.colors['text_light'],
                 bg=self.colors['card']).pack(pady=(2, 10))

        info_text = (
            "Đề tài 5 — Môn Học máy (Machine Learning)\n\n"
            "Ứng dụng demo nhận dạng chữ số viết tay (0–9)\n"
            "sử dụng các thuật toán K-NN và Decision Tree.\n\n"
            "Bộ dữ liệu: sklearn digits (1.797 ảnh 8×8)\n\n"
            "Công nghệ: Python, scikit-learn, Tkinter, matplotlib"
        )
        tk.Label(about_win,
                 text=info_text,
                 font=('Segoe UI', 9),
                 fg=self.colors['text'],
                 bg=self.colors['card'],
                 justify=tk.CENTER).pack(padx=20)

        tk.Button(about_win, text="Đóng",
                  font=('Segoe UI', 10),
                  bg=self.colors['primary'], fg='white',
                  relief='flat', padx=20, pady=4,
                  command=about_win.destroy).pack(pady=(15, 20))

    def _show_help(self):
        """Hiển thị hướng dẫn sử dụng"""
        help_win = tk.Toplevel(self.root)
        help_win.title("Hướng dẫn sử dụng")
        help_win.configure(bg=self.colors['card'])
        help_win.resizable(False, False)
        help_win.geometry("480x420")
        help_win.transient(self.root)

        tk.Label(help_win,
                 text="📖 Hướng dẫn sử dụng",
                 font=('Segoe UI', 14, 'bold'),
                 fg=self.colors['primary_dark'],
                 bg=self.colors['card']).pack(pady=(15, 10))

        help_text = (
            "1. Vẽ chữ số:\n"
            "   • Dùng chuột vẽ 1 hoặc nhiều chữ số (0–9) trên vùng trắng,\n"
            "     các chữ số viết CÁCH NHAU (vd: 2026)\n"
            "   • Điều chỉnh kích thước bút bằng thanh trượt\n"
            "   • Nhấn \"Nhận dạng\" để xem kết quả\n\n"
            "2. Tải ảnh:\n"
            "   • Nhấn \"Tải ảnh từ file\" để chọn ảnh chữ số\n"
            "   • Ảnh chụp chữ viết trên giấy: viết đậm, rõ, cách nhau;\n"
            "     chụp thẳng, đủ sáng, cắt bớt phần thừa nếu có\n"
            "   • Hỗ trợ: PNG, JPG, BMP, GIF, TIFF\n\n"
            "3. Mẫu ngẫu nhiên:\n"
            "   • Nhấn \"Mẫu ngẫu nhiên\" để xem model dự đoán\n"
            "     trên các mẫu từ tập kiểm tra\n\n"
            "4. Xóa canvas:\n"
            "   • Nhấn \"Xóa canvas\" hoặc Ctrl+D để xóa và vẽ lại\n\n"
            "Phím tắt: Ctrl+O (mở file), Ctrl+D (xóa canvas)"
        )
        tk.Label(help_win, text=help_text,
                 font=('Segoe UI', 9),
                 fg=self.colors['text'],
                 bg=self.colors['card'],
                 justify=tk.LEFT,
                 anchor='w').pack(padx=20, fill=tk.X)

        tk.Button(help_win, text="Đóng",
                  font=('Segoe UI', 10),
                  bg=self.colors['primary'], fg='white',
                  relief='flat', padx=20, pady=4,
                  command=help_win.destroy).pack(pady=(10, 15))


def main():
    """Hàm chính - khởi chạy ứng dụng"""
    root = tk.Tk()

    try:
        root.iconbitmap(default='')
    except Exception:
        pass

    app = DigitRecognizerApp(root)

    # Căn giữa cửa sổ
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'+{x}+{y}')

    print("[INFO] Ứng dụng đã sẵn sàng!")
    root.mainloop()


if __name__ == '__main__':
    main()
