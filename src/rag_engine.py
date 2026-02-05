import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class EnterpriseRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.vector_store = None
        # Lấy Key từ biến môi trường (Ưu tiên bảo mật)
        self.api_key = os.getenv("GOOGLE_API_KEY")

    def index_knowledge_base(self):
        """
        Hàm này quét thư mục 'data/' và các thư mục con (HR, IT...) 
        để nạp kiến thức vào vector database.
        """
        # 1. Dọn dẹp bộ nhớ cũ
        if os.path.exists(self.persist_directory):
            try: shutil.rmtree(self.persist_directory)
            except: pass

        if not os.path.exists("data"):
            os.makedirs("data")
            return "⚠️ Thư mục 'data' chưa tồn tại (đã tự động tạo). Hãy upload tài liệu vào đó."

        all_documents = []
        print("--- 🚀 BẮT ĐẦU QUÉT DỮ LIỆU ---")

        # 2. Quét thông minh: Hỗ trợ cả file ở gốc và trong thư mục con (Phân loại)
        # Ví dụ: data/HR/luong.txt -> category="HR"
        for root, dirs, files in os.walk("data"):
            category = os.path.basename(root) if root != "data" else "General"
            print(f"📂 Đang xử lý thư mục: {root} (Danh mục: {category})")
            
            # Load từng loại file trong thư mục hiện tại
            docs = []
            try: docs.extend(DirectoryLoader(root, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, silent_errors=True).load())
            except: pass
            try: docs.extend(DirectoryLoader(root, glob="*.pdf", loader_cls=PyPDFLoader, silent_errors=True).load())
            except: pass
            try: docs.extend(DirectoryLoader(root, glob="*.docx", loader_cls=Docx2txtLoader, silent_errors=True).load())
            except: pass

            # Gắn thẻ metadata (để sau này lọc nếu cần)
            for doc in docs:
                doc.metadata["category"] = category
            
            all_documents.extend(docs)

        if not all_documents:
            return "⚠️ Chưa tìm thấy tài liệu nào trong thư mục data."

        # 3. Cắt nhỏ văn bản (Chunking)
        # chunk_size=2000: Đủ lớn để chứa trọn vẹn một điều luật
        # chunk_overlap=200: Giữ mạch văn
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)

        # 4. Tạo Vector Store (ChromaDB)
        if self.api_key:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
            try:
                self.vector_store = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=self.persist_directory)
                return f"✅ THÀNH CÔNG: Đã học xong {len(all_documents)} tài liệu (chia thành {len(texts)} mảnh kiến thức)."
            except Exception as e:
                return f"❌ LỖI INDEXING: {str(e)}"
        else:
            return "❌ LỖI: Chưa có GOOGLE_API_KEY."

    def retrieve_answer(self, query, chat_history="", category=None):
        """
        Hàm trả lời câu hỏi với khả năng nhớ ngữ cảnh và định dạng Zalo.
        """
        if not self.api_key: return "Lỗi: Chưa cấu hình API Key."
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        
        # --- CẤU HÌNH AI "THÔNG MINH & TỰ NHIÊN" ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", # Dùng bản Pro để tư duy sâu
            google_api_key=self.api_key, 
            temperature=0.3,        # 0.3 giúp văn phong mềm mại nhưng vẫn chính xác
            max_output_tokens=8192, # Cho phép trả lời dài đầy đủ
            timeout=None,
            max_retries=2
        )
        
        # --- PROMPT CHUYÊN DỤNG CHO ZALO/MOBILE ---
        template = """Bạn là "Trợ lý HR Tận tâm" của Công ty Takagi Việt Nam.
        Nhiệm vụ: Hỗ trợ nhân viên giải đáp thắc mắc về quy định, chính sách, phúc lợi.

        LỊCH SỬ TRÒ CHUYỆN (Để hiểu ngữ cảnh):
        {chat_history}

        DỮ LIỆU TRA CỨU (Chỉ trả lời dựa trên thông tin này):
        {context}

        CÂU HỎI CỦA NHÂN VIÊN: "{question}"

        QUY TẮC TRẢ LỜI QUAN TRỌNG (ZALO FRIENDLY):
        1. **Định dạng:** Vì hiển thị trên điện thoại (Zalo), TUYỆT ĐỐI KHÔNG DÙNG BẢNG (NO TABLE).
           - Thay vào đó, hãy dùng danh sách gạch đầu dòng hoặc chia đoạn nhỏ.
           - Ví dụ: 
             * Mức A: 1.000.000 đ
             * Mức B: 2.000.000 đ
        2. **Thấu cảm:** Bắt đầu bằng giọng văn thân thiện, chia sẻ (đặc biệt với các vấn đề ốm đau, thai sản, kỷ luật).
        3. **Chính xác:** Trích dẫn số liệu cụ thể (tiền, ngày tháng, %) và ghi nguồn văn bản ở cuối.
        4. **Ngữ cảnh:** Nếu câu hỏi không rõ ràng (ví dụ "còn cái kia thì sao?"), hãy nhìn vào Lịch sử trò chuyện để hiểu.

        PHẢN HỒI CỦA BẠN:"""
        
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=template
        )
        
        # Cấu hình tìm kiếm
        search_kwargs = {"k": 6}
        # Nếu muốn lọc theo category (HR/IT), mở comment dòng dưới:
        # if category: search_kwargs["filter"] = {"category": category}

        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": QA_CHAIN_PROMPT,
                "memory": None
            }
        )
        
        # Thực thi và trả về kết quả
        try:
            result = qa_chain.invoke({"query": query, "chat_history": chat_history})
            return result["result"]
        except Exception as e:
            return f"Xin lỗi, hệ thống đang bận. Vui lòng thử lại sau. (Lỗi: {str(e)})"