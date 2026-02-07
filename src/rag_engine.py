import os
import shutil
import time

# --- CẤU HÌNH GOOGLE CHAT (REST) ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
import google.generativeai as genai

# Cấu hình Google GenAI
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"), transport="rest")

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
# SỬ DỤNG COHERE (Ổn định nhất cho gói Free)
from langchain_cohere import CohereEmbeddings

class EnterpriseRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.vector_store = None
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cohere_key = os.getenv("COHERE_API_KEY")
        
        # Cấu hình Cohere Embeddings
        if self.cohere_key:
            self.embedding_model = CohereEmbeddings(
                cohere_api_key=self.cohere_key,
                model="embed-multilingual-v3.0"
            )
        else:
            self.embedding_model = None

def index_knowledge_base(self):
        if not self.cohere_key: return "❌ Lỗi: Thiếu COHERE_API_KEY."

        # --- ĐOẠN CODE MỚI QUAN TRỌNG: GIẢI PHÓNG DB CŨ ---
        # Ngắt kết nối Chroma hiện tại để tránh lỗi "Readonly database"
        self.vector_store = None
        import gc
        gc.collect()
        # --------------------------------------------------

        # 1. Dọn dẹp DB cũ
        if os.path.exists(self.persist_directory):
            try: shutil.rmtree(self.persist_directory)
            except: pass # Nếu vẫn không xóa được thì bỏ qua, Chroma sẽ tự xử lý ghi đè

        if not os.path.exists("data"):
            os.makedirs("data")
            return "Folder data created."
            
        all_documents = []
        print("--- 🚀 START INDEXING WITH COHERE ---")
        
        # 2. Quét tài liệu
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
        
        # 3. Cắt nhỏ văn bản
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(all_documents)
        print(f"Tổng: {len(texts)} đoạn văn.")

        # 4. Lưu vào DB (Batching)
        try:
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
            batch_size = 20
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                self.vector_store.add_documents(batch)
                print(f"✅ Cohere: Xong lô {i//batch_size + 1}/{total_batches}")
                time.sleep(0.5) 
                
            return f"✅ Thành công! Đã học xong {len(all_documents)} tài liệu (Cohere Enterprise)."
            
        except Exception as e:
            return f"❌ Lỗi Indexing Cohere: {str(e)}"

    def retrieve_answer(self, query, chat_history="", category=None):
        if not self.api_key: return "Lỗi: Chưa cấu hình Google API Key."
        if not self.embedding_model: return "Lỗi: Chưa cấu hình Cohere API Key."
            
        self.vector_store = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embedding_model
        )
        
        # Model Chat (Google Gemini) - Tăng temp lên xíu cho tự nhiên
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=self.api_key, 
            temperature=0.3, 
            transport="rest"
        )
        
        # --- LOGIC TÌM KIẾM THÔNG MINH HƠN ---
        relevant_docs = []
        try:
            # 1. Thử tìm với category (nếu có)
            if category and category != "General":
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": 5, "filter": {"category": category}}
                )
                relevant_docs = retriever.invoke(query)
            
            # 2. Nếu không tìm thấy hoặc không có category, tìm toàn bộ DB
            if not relevant_docs:
                retriever_all = self.vector_store.as_retriever(search_kwargs={"k": 5})
                relevant_docs = retriever_all.invoke(query)
                
            # Nếu vẫn không có -> Hệ thống thực sự rỗng hoặc chưa Re-index
            if not relevant_docs:
                return "Dạ, hiện tại em chưa tìm thấy thông tin này trong hệ thống dữ liệu. Anh/chị kiểm tra lại xem đã cập nhật tài liệu (Re-index) chưa ạ?"
                
        except Exception as e:
            return f"Lỗi truy vấn DB: {str(e)}"
        
        # Xây dựng Context
        formatted_context = ""
        unique_sources = set()
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source_name", "Tài liệu nội bộ")
            unique_sources.add(source)
            content = doc.page_content.replace("\n", " ")
            formatted_context += f"[Nguồn {i+1}: {source}]\nNội dung: {content}\n\n"

        safe_history = chat_history.replace("{", "(").replace("}", ")")
        
        # --- PROMPT MỚI (CHUẨN PERSONA & TẬN TÂM) ---
        prompt = f"""
        VAI TRÒ:
        Bạn là Trợ lý HR ảo của công ty Takagi Việt Nam. Tên bạn là "Trợ lý HR".
        Bạn xưng hô là "em" và gọi người dùng là "anh/chị".
        Tính cách: Tận tâm, nhẹ nhàng, chuyên nghiệp nhưng gần gũi, luôn muốn giúp đỡ nhân viên.

        NHIỆM VỤ:
        Trả lời câu hỏi của nhân viên dựa trên DỮ LIỆU ĐƯỢC CUNG CẤP dưới đây.

        DỮ LIỆU TRA CỨU:
        {formatted_context}

        LỊCH SỬ TRÒ CHUYỆN:
        {safe_history}

        CÂU HỎI MỚI: "{query}"

        YÊU CẦU TRẢ LỜI:
        1. **Trung thực với dữ liệu:** Chỉ trả lời dựa trên thông tin trong phần "DỮ LIỆU TRA CỨU".
        2. **Xử lý khi thiếu tin:** Nếu dữ liệu không chứa câu trả lời, hãy nói: "Dạ, vấn đề này em tìm trong tài liệu nội bộ chưa thấy đề cập. Anh/chị liên hệ trực tiếp phòng Nhân sự để được hỗ trợ chính xác nhất nhé ạ."
        3. **Trích dẫn nguồn:** Cuối câu trả lời, hãy ghi chú nguồn tài liệu tham khảo. Ví dụ: (Theo: Noi_quy_cong_ty.pdf).
        4. **Phong cách:** - Bắt đầu bằng lời chào nhẹ nhàng nếu cần.
           - Giải thích rõ ràng, dễ hiểu.
           - **QUAN TRỌNG:** Luôn đưa ra một lời khuyên, đề xuất hoặc hành động tiếp theo ở cuối câu trả lời để hỗ trợ nhân viên tốt nhất (Ví dụ: "Anh/chị nhớ nộp đơn trước ngày 5 nhé", "Nếu cần mẫu đơn, anh/chị bảo em nha").

        BẮT ĐẦU TRẢ LỜI:
        """
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Lỗi Gemini: {str(e)}"