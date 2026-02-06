# 1. Dùng Python bản nhẹ nhất (Slim) để tải nhanh
FROM python:3.11-slim

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Copy file requirements trước (Để tận dụng Cache)
# Mẹo: Nếu file này không đổi, Docker sẽ bỏ qua bước cài đặt bên dưới -> Cực nhanh
COPY requirements.txt .

# 4. Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code vào sau
COPY . .

# 6. Mở cổng 8000
EXPOSE 8000

# 7. Lệnh chạy server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
