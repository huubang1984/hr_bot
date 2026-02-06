import os
import shutil

# --- CẤU HÌNH QUAN TRỌNG: ÉP DÙNG HTTP ĐỂ TRÁNH LỖI 0 ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import google.generativeai as genai
# Cấu hình thư viện Google chạy ở chế độ REST (HTTP) thay vì gRPC
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"), transport="rest")

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate

class EnterpriseRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.vector_store = None
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Sử dụng Model Embeddings 004 mới nhất
        if self.api_key:
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=self.api_key,
                # Thêm tham số này để LangChain cũng dùng REST
                transport="rest" 
            )
        else:
            self.embedding_model = None

    def index_knowledge_base(self):
        if not self.api_key: return "❌ Lỗi: Chưa có GOOGLE_API_KEY."

        # 1. Dọn dẹp DB cũ
        if os.path.exists(self.persist_directory):
            try: shutil.rmtree(self.persist_directory)
            except: pass

        if not os.path.exists("data"):
            os.makedirs("data")
            return "Folder data created."
            
        all_documents = []
        print("--- 🚀 START INDEXING (REST MODE) ---")
        
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

        # 4. Lưu vào Vector DB
        try:
            self.vector_store = Chroma.from_documents(
                documents=texts, 
                embedding=self.embedding_model,
                persist_directory=self.persist_directory
            )
            return f"✅ Thành công! Đã học xong {len(all_documents)} tài liệu."
        except Exception as e:
            # In lỗi chi tiết hơn
            return f"❌ Lỗi Indexing: {type(e).__name__} - {str(e)}"

    def retrieve_answer(self, query, chat_history="", category=None):
        if not self.api_key: return "Lỗi: Chưa cấu hình API Key."
            
        # Khởi tạo lại kết nối DB
        self.vector_store = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embedding_model
        )
        
        # Model Chat (Cũng ép dùng REST)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=self.api_key, 
            temperature=0.1,
            transport="rest"
        )
        
        # Tìm kiếm
        search_kwargs = {"k": 5}
        if category: search_kwargs["filter"] = {"category": category}

        try:
            retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            relevant_docs = retriever.invoke(query)
            
            # Kiểm tra nếu không tìm thấy gì (DB rỗng)
            if not relevant_docs:
                return "Hệ thống chưa có dữ liệu. Vui lòng chạy Re-index trước."
                
        except Exception as e:
            return f"Lỗi truy vấn DB: {str(e)}"
        
        # Xây dựng Context
        formatted_context = ""
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source_name", "Tài liệu nội bộ")
            content = doc.page_content.replace("\n", " ")
            formatted_context += f"[Nguồn {i+1}: {source}]\nNội dung: {content}\n\n"

        safe_history = chat_history.replace("{", "(").replace("}", ")")
        
        # Prompt
        prompt = f"""Bạn là Trợ lý HR của Takagi Việt Nam.
        
        DỮ LIỆU TRA CỨU:
        {formatted_context}
        ----------------
        LỊCH SỬ CHAT:
        {safe_history}
        ----------------
        CÂU HỎI: "{query}"
        
        YÊU CẦU:
        1. Trả lời ngắn gọn dựa trên dữ liệu tra cứu.
        2. Nếu không có thông tin, nói "Xin lỗi, không tìm thấy trong tài liệu".
        3. Ghi nguồn ở cuối câu trả lời.
        
        TRẢ LỜI:"""
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Lỗi Gemini: {str(e)}"