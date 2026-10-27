import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

# Tạo thư mục chứa ảnh
output_dir = "dataset_images"
os.makedirs(output_dir, exist_ok=True)

# Tạo các thư mục con từ 0 đến 9
for i in range(10):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

# Lưu từng ảnh vào thư mục tương ứng
print("Đang xuất ảnh...")
for idx, (img, label) in enumerate(zip(digits.images, digits.target)):
    # Đường dẫn lưu: dataset_images/<nhãn>/img_<thứ_tự>.png
    file_path = os.path.join(output_dir, str(label), f"digit_{idx}.png")
    
    # Lưu ảnh đen trắng
    plt.imsave(file_path, img, cmap="gray")

print(f"Đã lưu thành công toàn bộ 1.797 ảnh vào thư mục '{output_dir}'!")