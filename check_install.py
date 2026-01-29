import os
try:
    import langchain
    print("✅ TÌM THẤY LangChain tại:", langchain.__file__)
    
    # Kiểm tra xem folder 'chains' có thật sự tồn tại không
    lc_path = os.path.dirname(langchain.__file__)
    chains_path = os.path.join(lc_path, 'chains')
    
    if os.path.exists(chains_path):
        print("✅ Thư mục 'chains' ĐÃ CÓ MẶT.")
        import langchain.chains
        print("🚀 Import thành công! Bạn có thể chạy dự án.")
    else:
        print(f"❌ Vẫn KHÔNG thấy thư mục 'chains' tại: {chains_path}")
        print("👉 Nội dung thư mục langchain:", os.listdir(lc_path))
        
except ImportError as e:
    print("❌ Lỗi Import:", e)