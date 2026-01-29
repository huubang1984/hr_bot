import os
from src.rag_engine import EnterpriseRAG

def run_verification():
    print("\n--- KIỂM TRA HỆ THỐNG RAG (FINAL CHECK) ---")
    
    # 1. Nhập API Key
    api_key = input("👉 Nhập OpenAI API Key của bạn (sk-...): ").strip()
    if not api_key.startswith("sk-"):
        print("⚠️ Key có vẻ không đúng định dạng (phải bắt đầu bằng sk-). Nhưng tôi vẫn sẽ thử.")
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    rag = EnterpriseRAG()
    
    # 2. Test Indexing
    print("\n1. Đang Index dữ liệu...")
    status = rag.index_knowledge_base()
    print(status)
    
    if "THÀNH CÔNG" in status:
        # 3. Test Retrieval
        print("\n2. Đang thử hỏi AI: 'Quy trình xin nghỉ phép?'")
        try:
            answer = rag.retrieve_answer("Quy trình xin nghỉ phép như thế nào?")
            print("-" * 50)
            print(f"🤖 AI Trả lời:\n{answer}")
            print("-" * 50)
            print("🎉 CHÚC MỪNG! HỆ THỐNG ĐÃ HOẠT ĐỘNG HOÀN HẢO.")
        except Exception as e:
             print(f"❌ Lỗi khi hỏi AI: {e}")
             print("(Gợi ý: Kiểm tra xem API Key còn hạn mức (credit) không?)")
    else:
        print("❌ Dừng kiểm tra do lỗi Indexing.")

if __name__ == "__main__":
    run_verification()