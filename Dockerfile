# 1. Dùng Python bản nhẹ nhất (Slim) để tải nhanh
FROM python:3.11-slim

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 2b. Cài antiword để đọc file Word .doc cũ (định dạng OLE2)
RUN apt-get update \
    && apt-get install -y --no-install-recommends antiword \
    && rm -rf /var/lib/apt/lists/*

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
