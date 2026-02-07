import os
import time
# --- CẤU HÌNH GOOGLE CHAT (REST) ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
import google.generativeai as genai

if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"), transport="rest")

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
# --- THƯ VIỆN PINECONE MỚI ---
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

class EnterpriseRAG:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cohere_key = os.getenv("COHERE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Cấu hình Cohere Embeddings (1024 dimensions)
        if self.cohere_key:
            self.embedding_model = CohereEmbeddings(
                cohere_api_key=self.cohere_key,
                model="embed-multilingual-v3.0"
            )
        else:
            self.embedding_model = None

    def index_knowledge_base(self):
        if not self.cohere_key: return "❌ Lỗi: Thiếu COHERE_API_KEY."
        if not self.pinecone_api_key: return "❌ Lỗi: Thiếu PINECONE_API_KEY."
        if not self.index_name: return "❌ Lỗi: Thiếu PINECONE_INDEX_NAME."

        if not os.path.exists("data"):
            os.makedirs("data")
            return "Folder data created."
            
        all_documents = []
        print("--- 🚀 START INDEXING TO PINECONE CLOUD ---")
        
        # 1. Quét tài liệu
        for root, dirs, files in os.walk("data"):
            category = os.path.basename(root) if root != "data" else "General"
            docs = []
            try: docs.extend(DirectoryLoader(root, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, silent_errors=True).load())
            except: pass
            try: docs.extend(DirectoryLoader(root, glob="*.pdf", loader_cls=PyPDFLoader, silent_errors=True).load())
            except: pass
            try: docs.extend(DirectoryLoader(root, glob="*.docx", loader_cls=Docx2txtLoader, silent_errors=True).load())
            except: pass
            
            for doc in docs: 
                doc.metadata["category"] = category
                doc.metadata["source_name"] = os.path.basename(doc.metadata.get("source", ""))
            
            all_documents.extend(docs)

        if not all_documents: return "No documents found."
        
        # 2. Cắt nhỏ văn bản
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(all_documents)
        print(f"Tổng: {len(texts)} đoạn văn.")

        try:
            # 3. Kết nối Pinecone và Xóa dữ liệu cũ (Làm sạch Index)
            pc = Pinecone(api_key=self.pinecone_api_key)
            index = pc.Index(self.index_name)
            
            # Xóa toàn bộ vector cũ để nạp mới (Giống quy trình Re-index cũ)
            try:
                index.delete(delete_all=True)
                print("🗑️ Đã xóa dữ liệu cũ trên Cloud.")
                time.sleep(2) # Đợi Pinecone xử lý xóa
            except Exception as e:
                print(f"⚠️ Không thể xóa (có thể Index trống): {e}")

            # 4. Nạp dữ liệu mới (Batching)
            vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embedding_model,
                pinecone_api_key=self.pinecone_api_key
            )
            
            batch_size = 20
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                vector_store.add_documents(batch)
                print(f"☁️ Pinecone Upload: Xong lô {i//batch_size + 1}/{total_batches}")
                time.sleep(0.5) 
                
            return f"✅ Thành công! Đã đẩy {len(texts)} đoạn văn lên Mây (Pinecone)."
            
        except Exception as e:
            return f"❌ Lỗi Indexing Pinecone: {str(e)}"

    def retrieve_answer(self, query, chat_history="", category=None):
        if not self.api_key: return "Lỗi: Chưa cấu hình Google API Key."
        if not self.index_name: return "Lỗi: Chưa cấu hình Pinecone Index."
        
        vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embedding_model,
            pinecone_api_key=self.pinecone_api_key
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=self.api_key, 
            temperature=0.3, 
            transport="rest",
            max_output_tokens=8192 
        )
        
        relevant_docs = []
        try:
            if category and category != "General":
                retriever = vector_store.as_retriever(
                    search_kwargs={"k": 5, "filter": {"category": category}}
                )
                relevant_docs = retriever.invoke(query)
            
            if not relevant_docs:
                retriever_all = vector_store.as_retriever(search_kwargs={"k": 5})
                relevant_docs = retriever_all.invoke(query)
                
            if not relevant_docs:
                return "Dạ, hiện tại em chưa tìm thấy thông tin này trong hệ thống dữ liệu."
                
        except Exception as e:
            return f"Lỗi truy vấn Pinecone: {str(e)}"
        
        # --- SỬA ĐỔI 1: CÁCH FORMAT DỮ LIỆU ĐỂ AI ĐỌC TÊN FILE ---
        formatted_context = ""
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source_name", "Tài liệu nội bộ")
            content = doc.page_content.replace("\n", " ")
            # Bỏ chữ "Nguồn 1", ghi thẳng tên file để AI trích dẫn đúng
            formatted_context += f"--- TÀI LIỆU THAM KHẢO: {source} ---\nNội dung: {content}\n\n"

        safe_history = chat_history.replace("{", "(").replace("}", ")")
        
        # --- CẬP NHẬT PROMPT: Yêu cầu trả lời gọn và đầy đủ ---
        prompt = f"""
        VAI TRÒ:
        Bạn là Trợ lý HR ảo của công ty Takagi Việt Nam. Tên bạn là "Trợ lý HR".
        Bạn xưng hô là "em" và gọi người dùng là "anh/chị".
        Tính cách: Tận tâm, nhẹ nhàng, chuyên nghiệp nhưng gần gũi.
        
        DỮ LIỆU TRA CỨU:
        {formatted_context}

        LỊCH SỬ TRÒ CHUYỆN:
        {safe_history}

        CÂU HỎI MỚI: "{query}"

        YÊU CẦU TRẢ LỜI (QUAN TRỌNG):
        1. **Độ dài phù hợp:** Câu trả lời PHẢI ngắn gọn, súc tích (tối đa khoảng 1500 ký tự) để hiển thị tốt trên tin nhắn điện thoại. KHÔNG viết dài dòng lê thê.
        2. **Cấu trúc:** Sử dụng gạch đầu dòng (-) cho các ý chính để dễ đọc.
        3. **Trích dẫn chuẩn:** Tuyệt đối KHÔNG dùng "Nguồn 1, Nguồn 2". Hãy ghi rõ tên văn bản. 
           *Ví dụ đúng:* (Theo: Noi_quy_lao_dong_2024.pdf)
        4. **Xử lý nội dung dài:** Nếu nội quy quá dài, hãy tóm tắt các điểm quan trọng nhất và nói: "Nội dung chi tiết anh/chị xem thêm tại [Tên tài liệu] nhé ạ."
	5. Cuối cùng, đề xuất thêm gợi ý hoặc hỏi người dùng có cần thêm hỗ trợ nào khác không, "em" sẵn sàng hỗ trợ bất cứ lúc nào.

        BẮT ĐẦU TRẢ LỜI:
        """
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Lỗi Gemini: {str(e)}"