import os
import google.generativeai as genai

# 1. Nhập Key để kiểm tra
api_key = input("👉 Nhập Google API Key của bạn (AIza...): ").strip()
os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

print("\n--- DANH SÁCH MODEL BẠN ĐƯỢC DÙNG ---")
try:
    # Liệt kê tất cả model
    count = 0
    for m in genai.list_models():
        # Chỉ lấy những model hỗ trợ tạo nội dung (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ {m.name}")
            count += 1
            
    if count == 0:
        print("❌ Không tìm thấy model nào. Kiểm tra lại API Key hoặc vùng quốc gia.")
    else:
        print("\n💡 Hướng dẫn: Hãy copy chính xác phần tên (ví dụ 'gemini-1.5-flash') bỏ chữ 'models/' đi.")
        
except Exception as e:
    print(f"❌ Lỗi kết nối: {str(e)}")