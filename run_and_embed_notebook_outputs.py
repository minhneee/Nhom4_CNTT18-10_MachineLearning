"""
Chạy toàn bộ handwritten_digit_recognition.ipynb bằng Jupyter kernel thật (nbclient)
và lưu output (text, bảng, biểu đồ) vào chính file notebook.
Nếu bất kỳ cell nào lỗi, script DỪNG và báo lỗi (không nhúng output lỗi vào notebook).

Cách dùng:  python create_notebook.py && python run_and_embed_notebook_outputs.py
"""
import os
import nbformat
from nbclient import NotebookClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nb_path = os.path.join(BASE_DIR, 'handwritten_digit_recognition.ipynb')

nb = nbformat.read(nb_path, as_version=4)
NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': BASE_DIR}}).execute()
nbformat.write(nb, nb_path)
print(f"Đã chạy và lưu output cho {sum(c.cell_type == 'code' for c in nb.cells)} code cell: {nb_path}")
