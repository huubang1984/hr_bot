import os
import time
import glob
import google.generativeai as genai

# CẤU HÌNH
API_KEY = "AIzaSyC5GephZ338_CleQ7vlI7xpt7HkJ9Z6C6Y"
INPUT_FOLDER = "./scanned_pdfs" # Nơi để file PDF scan
OUTPUT_FOLDER = "./data"        # Nơi xuất file TXT cho RAG học

def ocr_with_gemini():
    # 1. Setup
    if not API_KEY:
        print("❌ Chưa có API Key. Hãy set biến môi trường GOOGLE_API_KEY")
        return
    
    genai.configure(api_key=API_KEY)
    
    # Tạo thư mục input nếu chưa có
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"📁 Đã tạo thư mục '{INPUT_FOLDER}'. Hãy copy file PDF scan vào đó rồi chạy lại!")
        return

    # 2. Quét file PDF
    pdf_files = glob.glob(os.path.join(INPUT_FOLDER, "*.pdf"))
    if not pdf_files:
        print(f"⚠️ Không tìm thấy file PDF nào trong thư mục '{INPUT_FOLDER}'.")
        return

    print(f"🔍 Tìm thấy {len(pdf_files)} file scan. Bắt đầu xử lý với Gemini Vision...")
    
    # Model Gemini Flash hỗ trợ đọc tài liệu rất tốt và rẻ
    model = genai.GenerativeModel("gemini-1.5-flash") 

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\n📄 Đang xử lý: {filename}...")
        
        try:
            # A. Upload file lên Google Server (tạm thời)
            print("   -> Đang upload file lên Gemini...")
            sample_file = genai.upload_file(path=pdf_path, display_name=filename)
            
            # Đợi file sẵn sàng
            while sample_file.state.name == "PROCESSING":
                print("   -> Đang xử lý ảnh...", end="\r")
                time.sleep(2)
                sample_file = genai.get_file(sample_file.name)
            
            # B. Yêu cầu AI đọc
            print("   -> Đang OCR (Nhận diện chữ)...")
            response = model.generate_content([
                "Hãy trích xuất toàn bộ văn bản trong tài liệu này. Giữ nguyên định dạng tiếng Việt.", 
                sample_file
            ])
            
            # C. Lưu kết quả ra file .txt trong folder data
            txt_filename = filename.replace(".pdf", ".txt")
            output_path = os.path.join(OUTPUT_FOLDER, txt_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)
                
            print(f"✅ XONG! Đã lưu tại: {output_path}")
            
            # D. Dọn dẹp file trên cloud
            genai.delete_file(sample_file.name)
            
        except Exception as e:
            print(f"❌ Lỗi file {filename}: {str(e)}")

if __name__ == "__main__":
    ocr_with_gemini()