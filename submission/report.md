# Báo Cáo Lab: CI/CD cho AI Systems

**Course:** AIInAction - VinUni | **Buổi:** Day 21

---

## 1. Bộ Siêu Tham Số Đã Chọn

Sau khi chạy 3 thí nghiệm với MLflow, bộ tham số cuối cùng được chọn:

| Tham số | Giá trị |
|---|---|
| `model_type` | `random_forest` |
| `n_estimators` | 500 |
| `max_depth` | 15 |
| `min_samples_split` | 2 |

**Kết quả so sánh 3 lần chạy:**

| Lần | n_estimators | max_depth | Accuracy |
|---|---|---|---|
| 1 | 100 | 5 | 0.5640 |
| 2 | 200 | 10 | 0.6480 |
| 3 | 500 | 15 | **0.7440** |

**Lý do chọn:** Accuracy tăng rõ rệt theo số lượng cây và độ sâu. Lần 3 đạt 0.7440, vượt ngưỡng yêu cầu 0.70. `min_samples_split=2` giữ nguyên để mỗi nút có thể phân chia linh hoạt nhất.

---

## 2. Khó Khăn Gặp Phải và Cách Giải Quyết

**Lỗi cài đặt pip trên Ubuntu mới (externally managed environment)**
- Hệ điều hành Ubuntu 24.04 chặn `pip install` trực tiếp vào system Python.
- Giải quyết: tạo virtual environment `python3 -m venv /root/venv` và cài package vào đó. Cập nhật systemd service trỏ `ExecStart` sang `/root/venv/bin/python`.

**Deploy job fail do health check chạy quá sớm**
- Service cần tải model từ S3 khi khởi động, mất hơn 5 giây. `sleep 5` cố định không đủ.
- Giải quyết: thay bằng retry loop kiểm tra mỗi 5 giây, tối đa 60 giây.

**Workflow không trigger khi thay đổi file `.github/workflows/mlops.yml`**
- `paths` filter không bao gồm chính file workflow.
- Giải quyết: thêm `'.github/workflows/mlops.yml'` vào danh sách `paths`.

**MLflow lỗi `MissingConfigException` khi chạy local**
- Thư mục `mlruns/` bị corrupt do quá trình chạy bị gián đoạn.
- Giải quyết: xóa `mlruns/` và chạy lại.

**Accuracy dưới ngưỡng 0.70 với params mặc định**
- Các params ban đầu (`n_estimators=100, max_depth=5`) cho accuracy 0.564.
- Giải quyết: tăng dần params qua 3 thí nghiệm, chọn bộ tốt nhất đạt 0.744.
