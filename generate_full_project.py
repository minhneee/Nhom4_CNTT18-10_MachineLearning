"""
Script tự động chạy toàn bộ thí nghiệm học máy cho Đề tài 5:
- ĐỌC VÀ HUẤN LUYỆN TRỰC TIẾP TỪ THƯ MỤC ẢNH: dataset_images/ (theo đúng chuẩn bài toán Thị giác máy tính)
- Tiền xử lý ảnh: grayscale, chuẩn hóa pixel [0, 1] (/255.0)
- Trích xuất đặc trưng: Raw Pixels, HOG, PCA
- Huấn luyện K-NN và Decision Tree
- Lưu các biểu đồ chất lượng cao vào thư mục figures/
- Lưu mô hình tốt nhất best_model.pkl
- Xuất bảng kết quả chi tiết model_comparison_results.csv
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from skimage.feature import hog
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset_images')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

print("=== BẮT ĐẦU CHẠY PIPELINE HUẤN LUYỆN TỪ THƯ MỤC dataset_images ===")

# 1. Đọc dữ liệu ảnh trực tiếp từ thư mục dataset_images/
image_paths = []
images = []
labels = []

classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"[1/8] Tìm thấy {len(classes)} thư mục lớp: {classes}")

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for f in files:
        fpath = os.path.join(class_dir, f)
        # Mở ảnh bằng PIL và chuyển sang grayscale
        img = Image.open(fpath).convert('L')
        # Chuẩn hóa về [0, 1] qua phép chia 255.0
        arr = np.array(img, dtype=np.float32) / 255.0
        images.append(arr.flatten())
        labels.append(int(class_name))
        image_paths.append(fpath)

X = np.array(images)
y = np.array(labels)
print(f"[1/8] Đã nạp thành công {len(X)} ảnh từ dataset_images. Kích thước vector: {X.shape[1]} đặc trưng (8x8).")
print(f"      Giá trị NaN: {np.isnan(X).sum()} | Số ảnh trùng lặp hoàn toàn: {len(X) - len(np.unique(X, axis=0))}")

# 2. Phân chia Train / Test (80/20, Stratified, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[2/8] Chia tập dữ liệu: Train = {X_train.shape[0]} ảnh, Test = {X_test.shape[0]} ảnh.")

# 3. Trích xuất đặc trưng
# Feature 1: Raw Pixels (64 chiều)
# Feature 2: HOG (2x2 cells, 8 orientations -> 128 chiều)
print("[3/8] Đang trích xuất đặc trưng HOG...")
X_train_hog = np.array([
    hog(img.reshape(8, 8), orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1))
    for img in X_train
])
X_test_hog = np.array([
    hog(img.reshape(8, 8), orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1))
    for img in X_test
])

# Feature 3: PCA (Fit trên Train duy nhất)
print("[3/8] Đang áp dụng PCA (giữ 95% phương sai)...")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
n_pca_components = X_train_pca.shape[1]
print(f"      PCA chọn {n_pca_components} thành phần chính (giữ {pca.explained_variance_ratio_.sum()*100:.2f}% phương sai).")

# 4. Trực quan hóa EDA & lưu hình ảnh
print("[4/8] Đang vẽ và lưu các biểu đồ EDA...")

# 4.1 Phân phối nhãn
plt.figure(figsize=(9, 4.5))
sns.countplot(x=y, palette='Blues_r')
plt.title('Phân phối số lượng mẫu giữa các lớp trong dataset_images (0 - 9)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Chữ số (Lớp)', fontsize=11)
plt.ylabel('Số lượng ảnh', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i in range(10):
    count = (y == i).sum()
    plt.text(i, count + 2, str(count), ha='center', fontsize=10, fontweight='bold')
plt.ylim(0, 210)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_label_distribution.png'), dpi=300)
plt.close()

# 4.2 Mẫu chữ số tiêu biểu mỗi lớp
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y == i)[0][0]
    ax.imshow(X[idx].reshape(8, 8), cmap='gray_r', interpolation='nearest')
    ax.set_title(f'Lớp: {i}', fontsize=12, fontweight='bold')
    ax.axis('off')
plt.suptitle('Ảnh chữ số thực tế đọc từ dataset_images/ (8x8 pixel)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_sample_digits.png'), dpi=300)
plt.close()

# 4.3 Ảnh trung bình (Mean Digit Image)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    mean_img = X[y == i].mean(axis=0).reshape(8, 8)
    ax.imshow(mean_img, cmap='magma', interpolation='nearest')
    ax.set_title(f'Mean: {i}', fontsize=12, fontweight='bold')
    ax.axis('off')
plt.suptitle('Ảnh trung bình (Mean Image) của từng lớp chữ số', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_mean_digits.png'), dpi=300)
plt.close()

# 4.4 Biểu đồ Explained Variance của PCA
pca_full = PCA().fit(X_train)
cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
plt.figure(figsize=(8, 4.5))
plt.plot(range(1, len(cum_var) + 1), cum_var, 'b-o', markersize=3, label='Phương sai tích lũy')
plt.axhline(y=95, color='r', linestyle='--', label='Ngưỡng 95% phương sai')
plt.axvline(x=n_pca_components, color='g', linestyle=':', label=f'n_components={n_pca_components}')
plt.title('Đường cong phương sai tích lũy PCA (Tập huấn luyện)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Số thành phần chính', fontsize=11)
plt.ylabel('Tỉ lệ phương sai tích lũy (%)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'pca_explained_variance.png'), dpi=300)
plt.close()

# 5. Huấn luyện và Đánh giá mô hình
print("[5/8] Đang thực hiện các thí nghiệm học máy...")
results_list = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, feat_name, params_str):
    # CV chỉ dùng tập Train -> dùng để chọn siêu tham số
    cv_res = cross_validate(model, X_tr, y_tr, cv=cv, scoring=['accuracy', 'f1_macro'], n_jobs=1)

    # Thời gian huấn luyện = thời gian fit 1 lần trên toàn bộ tập Train
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_time = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    preds = model.predict(X_te)
    predict_time = (time.perf_counter() - t0) * 1000
    
    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, average='macro')
    rec = recall_score(y_te, preds, average='macro')
    f1 = f1_score(y_te, preds, average='macro')
    
    return {
        'Mô hình': name,
        'Biểu diễn đặc trưng': feat_name,
        'Tham số': params_str,
        'CV Accuracy': f"{cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}",
        'CV F1-Macro': f"{cv_res['test_f1_macro'].mean():.4f} ± {cv_res['test_f1_macro'].std():.4f}",
        'Test Accuracy': acc,
        'Test Precision': prec,
        'Test Recall': rec,
        'Test F1-Macro': f1,
        'Train Time (ms)': round(train_time, 2),
        'Predict Time (ms)': round(predict_time, 2),
        '_cv_acc': cv_res['test_accuracy'].mean(),
        '_model_obj': model,
        '_preds': preds
    }

# Baseline: K-NN (k=1, Euclidean) trên Raw pixels
baseline_knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
res_baseline = evaluate_model('Baseline: K-NN (k=1)', baseline_knn, X_train, X_test, y_train, y_test, 'Raw Pixels (64)', 'k=1, metric=euclidean')
results_list.append(res_baseline)

# K-NN Tuning trên Raw Pixels
k_values = [1, 3, 5, 7, 9, 11, 15]
metrics = ['euclidean', 'manhattan']
knn_cv = {}  # (k, metric) -> CV accuracy (chỉ dùng tập Train để chọn tham số)

for k in k_values:
    for metric in metrics:
        m = KNeighborsClassifier(n_neighbors=k, metric=metric)
        r = evaluate_model('K-NN', m, X_train, X_test, y_train, y_test, 'Raw Pixels (64)', f'k={k}, metric={metric}')
        results_list.append(r)
        knn_cv[(k, metric)] = r['_cv_acc']

# Best KNN chọn theo CV accuracy (KHÔNG dùng tập Test để chọn)
best_k, best_metric = max(knn_cv, key=knn_cv.get)
print(f"      Best K-NN theo CV: k={best_k}, metric={best_metric} (CV acc = {knn_cv[(best_k, best_metric)]:.4f})")
best_knn_raw = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric).fit(X_train, y_train)

# K-NN trên HOG và PCA với tham số tốt nhất
results_list.append(evaluate_model('K-NN', KNeighborsClassifier(n_neighbors=best_k, metric=best_metric),
                                   X_train_hog, X_test_hog, y_train, y_test, 'HOG (128)', f'k={best_k}, metric={best_metric}'))
results_list.append(evaluate_model('K-NN', KNeighborsClassifier(n_neighbors=best_k, metric=best_metric),
                                   X_train_pca, X_test_pca, y_train, y_test, f'PCA ({n_pca_components})', f'k={best_k}, metric={best_metric}'))

# Decision Tree Tuning trên Raw Pixels: max_depth x criterion
dt_depths = [3, 5, 8, 10, 15, None]
criteria = ['gini', 'entropy']
dt_cv, dt_train_acc = {}, {}  # (depth, crit) -> accuracy

for crit in criteria:
    for d in dt_depths:
        dt = DecisionTreeClassifier(max_depth=d, criterion=crit, random_state=42)
        r = evaluate_model('Decision Tree', dt, X_train, X_test, y_train, y_test, 'Raw Pixels (64)', f'depth={d}, crit={crit}')
        results_list.append(r)
        dt_cv[(d, crit)] = r['_cv_acc']
        dt_train_acc[(d, crit)] = accuracy_score(y_train, dt.predict(X_train))

# Best Decision Tree chọn theo CV accuracy
best_depth, best_crit = max(dt_cv, key=dt_cv.get)
print(f"      Best Decision Tree theo CV: depth={best_depth}, criterion={best_crit} (CV acc = {dt_cv[(best_depth, best_crit)]:.4f})")
best_dt = DecisionTreeClassifier(max_depth=best_depth, criterion=best_crit, random_state=42).fit(X_train, y_train)

# Decision Tree trên HOG & PCA với tham số tốt nhất
for X_tr_f, X_te_f, feat in [(X_train_hog, X_test_hog, 'HOG (128)'), (X_train_pca, X_test_pca, f'PCA ({n_pca_components})')]:
    results_list.append(evaluate_model('Decision Tree', DecisionTreeClassifier(max_depth=best_depth, criterion=best_crit, random_state=42),
                                       X_tr_f, X_te_f, y_train, y_test, feat, f'depth={best_depth}, crit={best_crit}'))

print("[6/8] Đang xuất các biểu đồ đánh giá...")

# 5.1 Biểu đồ K-NN Tuning
plt.figure(figsize=(8, 4.5))
plt.plot(k_values, [knn_cv[(k, 'euclidean')] for k in k_values], 'o-', color='#1f77b4', linewidth=2, label='Khoảng cách Euclidean')
plt.plot(k_values, [knn_cv[(k, 'manhattan')] for k in k_values], 's--', color='#ff7f0e', linewidth=2, label='Khoảng cách Manhattan')
plt.title('Ảnh hưởng của k và hàm khoảng cách lên K-NN (5-Fold CV trên tập Train)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Số láng giềng k', fontsize=11)
plt.ylabel('Độ chính xác CV (trung bình 5 fold)', fontsize=11)
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'knn_k_tuning.png'), dpi=300)
plt.close()

# 5.2 Biểu đồ Overfitting Decision Tree
depth_labels = [str(d) for d in dt_depths]
plt.figure(figsize=(8.5, 4.5))
plt.plot(depth_labels, [dt_train_acc[(d, best_crit)] for d in dt_depths], 'o-', color='#2ca02c', label=f'Train Accuracy ({best_crit})')
plt.plot(depth_labels, [dt_cv[(d, 'gini')] for d in dt_depths], 's--', color='#1f77b4', label='CV Accuracy (gini)')
plt.plot(depth_labels, [dt_cv[(d, 'entropy')] for d in dt_depths], 'o--', color='#d62728', label='CV Accuracy (entropy)')
plt.title('Hiện tượng Overfitting ở Cây quyết định theo độ sâu (max_depth)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Độ sâu tối đa (max_depth)', fontsize=11)
plt.ylabel('Độ chính xác (Accuracy)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'dt_overfitting_curve.png'), dpi=300)
plt.close()

# 5.3 Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cm_base = confusion_matrix(y_test, baseline_knn.predict(X_test))
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title(f'Baseline: K-NN (k=1)\nAccuracy: {accuracy_score(y_test, baseline_knn.predict(X_test)):.4f}', fontweight='bold')
axes[0].set_xlabel('Dự đoán (Predicted)')
axes[0].set_ylabel('Thực tế (Actual)')

cm_knn = confusion_matrix(y_test, best_knn_raw.predict(X_test))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title(f'Best K-NN (k={best_k}, {best_metric})\nAccuracy: {accuracy_score(y_test, best_knn_raw.predict(X_test)):.4f}', fontweight='bold')
axes[1].set_xlabel('Dự đoán (Predicted)')
axes[1].set_ylabel('Thực tế (Actual)')

cm_dt = confusion_matrix(y_test, best_dt.predict(X_test))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', ax=axes[2], cbar=False)
axes[2].set_title(f'Best Decision Tree (depth={best_depth}, {best_crit})\nAccuracy: {accuracy_score(y_test, best_dt.predict(X_test)):.4f}', fontweight='bold')
axes[2].set_xlabel('Dự đoán (Predicted)')
axes[2].set_ylabel('Thực tế (Actual)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix_comparison.png'), dpi=300)
plt.close()

# 5.4 ROC Curves
y_test_bin = label_binarize(y_test, classes=range(10))
y_score_knn = best_knn_raw.predict_proba(X_test)
plt.figure(figsize=(9, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(10):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_knn[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=1.8, label=f'Chữ số {i} (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Ngẫu nhiên (AUC = 0.5)')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=11)
plt.ylabel('True Positive Rate (TPR)', fontsize=11)
plt.title('Đường cong ROC One-vs-Rest (Best K-NN trên dataset_images)', fontsize=13, fontweight='bold', pad=12)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'roc_curves_best_knn.png'), dpi=300)
plt.close()

# 6. Error Analysis
print("[7/8] Đang thực hiện phân tích lỗi...")
preds_best = best_knn_raw.predict(X_test)
misclassified_idx = np.where(preds_best != y_test)[0]
print(f"      Số mẫu dự đoán sai: {len(misclassified_idx)} / {len(y_test)} ({len(misclassified_idx)/len(y_test)*100:.2f}%)")

if len(misclassified_idx) > 0:
    # Hiển thị TẤT CẢ mẫu sai (tối đa 5 cột mỗi hàng)
    n_mis = len(misclassified_idx)
    n_cols = min(n_mis, 5)
    n_rows = int(np.ceil(n_mis / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4.0 * n_rows), squeeze=False,
                             gridspec_kw={'hspace': 0.45})
    for ax in axes.flat:
        ax.axis('off')
    for ax, idx in zip(axes.flat, misclassified_idx):
        ax.imshow(X_test[idx].reshape(8, 8), cmap='gray_r', interpolation='nearest')
        ax.set_title(f'Thực tế: {y_test[idx]}\nDự đoán: {preds_best[idx]}', fontsize=12, color='red', fontweight='bold')
    print("      Các cặp nhầm lẫn (thực tế -> dự đoán):",
          sorted(zip(y_test[misclassified_idx].tolist(), preds_best[misclassified_idx].tolist())))
    plt.suptitle(f'Toàn bộ {n_mis} mẫu bị K-NN (k={best_k}) dự đoán sai trên tập Test', fontsize=13, fontweight='bold')
    plt.savefig(os.path.join(FIG_DIR, 'error_analysis_misclassified.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 6b. Phân tích bổ sung (dùng trong báo cáo)
print("[7b] Đang thực hiện các phân tích bổ sung...")
os.makedirs(os.path.join(BASE_DIR, 'report'), exist_ok=True)
vn1 = lambda x, d=2: f"{x:.{d}f}".replace('.', ',')

# (a) Thống kê điểm ảnh: ảnh trung bình và độ lệch chuẩn của từng vị trí trên tập huấn luyện
pix_std = X_train.std(axis=0)
print(f"      Điểm ảnh có phương sai 0: {(pix_std == 0).sum()} (vị trí {np.where(pix_std == 0)[0].tolist()}), "
      f"phương sai < 0,001: {(pix_std ** 2 < 1e-3).sum()}; tỉ lệ điểm ảnh bằng 0: {(X == 0).mean():.3f}")
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, img, title in [(axes[0], X_train.mean(axis=0), 'Giá trị trung bình'), (axes[1], pix_std, 'Độ lệch chuẩn')]:
    im = ax.imshow(img.reshape(8, 8), cmap='viridis', interpolation='nearest')
    for (r, c), v in np.ndenumerate(img.reshape(8, 8)):
        ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7, color='white' if v < img.max() * 0.6 else 'black')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_pixel_stats.png'), dpi=300)
plt.close()

# (b) Đường cong học: CV accuracy theo kích thước tập huấn luyện (chỉ dùng tập Train)
sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
lc = {}
for name, m in [('K-NN', KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)),
                ('Cây quyết định', DecisionTreeClassifier(max_depth=best_depth, criterion=best_crit, random_state=42))]:
    n, tr, va = learning_curve(m, X_train, y_train, train_sizes=sizes, cv=cv, n_jobs=1)
    lc[name] = (n, tr.mean(axis=1), va.mean(axis=1))
plt.figure(figsize=(8, 4.5))
for (name, (n, tr, va)), color in zip(lc.items(), ['#1f77b4', '#d62728']):
    plt.plot(n, tr, 'o--', color=color, alpha=0.6, label=f'{name}: Train')
    plt.plot(n, va, 'o-', color=color, linewidth=2, label=f'{name}: CV')
plt.title('Đường cong học (5-Fold CV trên tập huấn luyện)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Số mẫu dùng để huấn luyện', fontsize=11)
plt.ylabel('Độ chính xác', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'learning_curve.png'), dpi=300)
plt.close()
with open(os.path.join(BASE_DIR, 'report', 'learning_curve.tex'), 'w', encoding='utf-8') as f:
    n_knn, tr_knn, va_knn = lc['K-NN']
    _, tr_dt, va_dt = lc['Cây quyết định']
    for i in range(len(n_knn)):
        f.write(f"{n_knn[i]} & {vn1(tr_knn[i]*100)} & {vn1(va_knn[i]*100)} & {vn1(tr_dt[i]*100)} & {vn1(va_dt[i]*100)} \\\\\n")
print("      Đường cong học (CV %):", {k: np.round(v[2] * 100, 2).tolist() for k, v in lc.items()})

# (c) Cấu trúc cây quyết định tốt nhất và mức độ quan trọng của từng điểm ảnh
imp = best_dt.feature_importances_
print(f"      Cây tốt nhất: độ sâu {best_dt.get_depth()}, {best_dt.get_n_leaves()} lá, "
      f"{(imp == 0).sum()}/64 điểm ảnh không được dùng; 5 điểm ảnh quan trọng nhất: "
      f"{[(int(i), round(float(imp[i]), 3)) for i in np.argsort(imp)[::-1][:5]]}")
plt.figure(figsize=(4.8, 4.2))
plt.imshow(imp.reshape(8, 8), cmap='Reds', interpolation='nearest')
plt.colorbar(fraction=0.046)
plt.title('Mức độ quan trọng của điểm ảnh\n(Cây quyết định tốt nhất)', fontsize=11, fontweight='bold')
plt.xticks(range(8)); plt.yticks(range(8))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'dt_feature_importance.png'), dpi=300)
plt.close()

# (d) Các cặp nhầm lẫn nhiều nhất của cây quyết định
cm_dt_off = confusion_matrix(y_test, best_dt.predict(X_test))
np.fill_diagonal(cm_dt_off, 0)
top_pairs = sorted(((cm_dt_off[i, j], i, j) for i in range(10) for j in range(10) if cm_dt_off[i, j] > 0), reverse=True)[:6]
with open(os.path.join(BASE_DIR, 'report', 'dt_confusions.tex'), 'w', encoding='utf-8') as f:
    for cnt, i, j in top_pairs:
        f.write(f"{i} & {j} & {cnt} & {vn1(cnt / cm_dt_off.sum() * 100, 1)} \\\\\n")
print(f"      Cây quyết định sai {cm_dt_off.sum()} mẫu; các cặp nhầm nhiều nhất: {[(int(i), int(j), int(c)) for c, i, j in top_pairs]}")

# (e) Độ tin cậy của K-NN (tỉ lệ phiếu bầu) và lựa chọn ngưỡng chuyển kiểm tra thủ công
proba_best = best_knn_raw.predict_proba(X_test)
conf = proba_best.max(axis=1)
with open(os.path.join(BASE_DIR, 'report', 'knn_confidence.tex'), 'w', encoding='utf-8') as f:
    for thr, label in [(0.0, 'Không dùng ngưỡng'), (0.6, '$\\ge 2/3$ (ít nhất 2/3 láng giềng đồng thuận)'),
                       (0.99, '$= 1$ (cả 3 láng giềng đồng thuận)')]:
        keep = conf >= thr
        acc_keep = (preds_best[keep] == y_test[keep]).mean()
        f.write(f"{label} & {keep.sum()} ({vn1(keep.mean()*100, 1)}\\%) & {vn1(acc_keep*100)} & {(~keep).sum()} \\\\\n")
        print(f"      Ngưỡng {thr}: xử lý tự động {keep.sum()}/{len(keep)} mẫu, độ chính xác {acc_keep:.4f}")
nn_dist = best_knn_raw.kneighbors(X_test, n_neighbors=1)[0][:, 0]
wrong = preds_best != y_test
print(f"      Khoảng cách tới láng giềng gần nhất: mẫu đúng {nn_dist[~wrong].mean():.3f}, mẫu sai {nn_dist[wrong].mean():.3f}; "
      f"số mẫu sai mà cả 3 láng giềng đồng thuận: {(conf[wrong] > 0.99).sum()}/{wrong.sum()}")

# 7. Lưu mô hình tốt nhất
print("[8/8] Đang lưu best_model.pkl...")
best_model_path = os.path.join(BASE_DIR, 'best_model.pkl')
joblib.dump(best_knn_raw, best_model_path)
print(f"      Đã lưu: {best_model_path}")

# Bảng so sánh
summary_df = pd.DataFrame(results_list).drop(columns=['_cv_acc', '_model_obj', '_preds'])
summary_df_path = os.path.join(BASE_DIR, 'model_comparison_results.csv')
summary_df.to_csv(summary_df_path, index=False, encoding='utf-8-sig')
print(f"      Đã lưu bảng so sánh: {summary_df_path}")

# Xuất bảng kết quả dạng LaTeX cho báo cáo -> số liệu trong báo cáo luôn khớp lần chạy thật.
# Định dạng số kiểu Việt Nam: dấu phẩy thập phân, đơn vị %.
os.makedirs(os.path.join(BASE_DIR, 'report'), exist_ok=True)

def vn(x, d=2):
    return f"{x:.{d}f}".replace('.', ',')

def latex_row(r, label=None):
    cv_mean, cv_std = (float(v) * 100 for v in r['CV Accuracy'].split(' ± '))
    model = 'Cây quyết định' if r['Mô hình'] == 'Decision Tree' else 'K-NN'
    feat = r['Biểu diễn đặc trưng'].replace('Raw Pixels', 'Raw')
    params = r['Tham số'].replace('metric=', '').replace('crit=', '').replace('depth=', 'max\\_depth=')
    cells = [label or model, feat, params, f"{vn(cv_mean)} $\\pm$ {vn(cv_std)}",
             vn(r['Test Accuracy'] * 100), vn(r['Test Precision'] * 100), vn(r['Test Recall'] * 100),
             vn(r['Test F1-Macro'] * 100), vn(r['Train Time (ms)']), vn(r['Predict Time (ms)'])]
    return ' & '.join(cells) + ' \\\\\n'

# Bảng đầy đủ (phụ lục): tất cả cấu hình đã chạy
with open(os.path.join(BASE_DIR, 'report', 'results_table.tex'), 'w', encoding='utf-8') as f:
    for i, (_, r) in enumerate(summary_df.iterrows()):
        f.write(latex_row(r, 'Baseline K-NN' if i == 0 else None))

# Bảng rút gọn (chương 4): baseline + cấu hình tốt nhất của mỗi thuật toán trên 3 cách biểu diễn
def pick(model, feat_prefix, params):
    m = summary_df[(summary_df['Mô hình'] == model) & summary_df['Biểu diễn đặc trưng'].str.startswith(feat_prefix)
                   & (summary_df['Tham số'] == params)]
    return m.iloc[0]

knn_p, dt_p = f'k={best_k}, metric={best_metric}', f'depth={best_depth}, crit={best_crit}'
main_rows = [(summary_df.iloc[0], 'Baseline K-NN')] + \
            [(pick('K-NN', feat, knn_p), 'K-NN') for feat in ('Raw', 'HOG', 'PCA')] + \
            [(pick('Decision Tree', feat, dt_p), 'Cây quyết định') for feat in ('Raw', 'HOG', 'PCA')]
with open(os.path.join(BASE_DIR, 'report', 'results_main.tex'), 'w', encoding='utf-8') as f:
    for j, (r, label) in enumerate(main_rows):
        if j in (1, 4):
            f.write('\\midrule\n')
        f.write(latex_row(r, label))
print("      Đã xuất bảng LaTeX: report/results_table.tex, report/results_main.tex")

# Bảng Precision / Recall / F1 theo từng lớp: Best K-NN vs Best Decision Tree
rep_knn = classification_report(y_test, preds_best, output_dict=True)
rep_dt = classification_report(y_test, best_dt.predict(X_test), output_dict=True)
with open(os.path.join(BASE_DIR, 'report', 'per_class_table.tex'), 'w', encoding='utf-8') as f:
    for c in [str(i) for i in range(10)]:
        a, b = rep_knn[c], rep_dt[c]
        f.write(f"{c} & {int(a['support'])} & {vn(a['precision']*100)} & {vn(a['recall']*100)} & {vn(a['f1-score']*100)} & "
                f"{vn(b['precision']*100)} & {vn(b['recall']*100)} & {vn(b['f1-score']*100)} \\\\\n")
print("      Đã xuất bảng LaTeX: report/per_class_table.tex")

# Cập nhật ảnh sang thư mục report/figures/
report_fig_dir = os.path.join(BASE_DIR, 'report', 'figures')
if os.path.exists(report_fig_dir):
    import shutil
    for fname in os.listdir(FIG_DIR):
        if fname.endswith('.png'):
            shutil.copy(os.path.join(FIG_DIR, fname), os.path.join(report_fig_dir, fname))
    print(f"      Đã đồng bộ hình ảnh sang: {report_fig_dir}")

print("=== HOÀN TẤT HUẤN LUYỆN TRỰC TIẾP TỪ dataset_images ===")
