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
        self.api_key = os.getenv("GOOGLE_API_KEY")

    def index_knowledge_base(self):
        # ... (Giữ nguyên phần Indexing không đổi) ...
        # (Để ngắn gọn, tôi không paste lại phần Indexing dài dòng ở đây vì nó vẫn hoạt động tốt)
        if os.path.exists(self.persist_directory):
            try: shutil.rmtree(self.persist_directory)
            except: pass

        if not os.path.exists("data"):
            os.makedirs("data")
            return "⚠️ Thư mục 'data' chưa tồn tại."

        all_documents = []
        print("--- 🚀 BẮT ĐẦU QUÉT DỮ LIỆU ---")
        for root, dirs, files in os.walk("data"):
            category = os.path.basename(root) if root != "data" else "General"
            docs = []
            try: docs.extend(DirectoryLoader(root, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, silent_errors=True).load())
            except: pass
            try: docs.extend(DirectoryLoader(root, glob="*.pdf", loader_cls=PyPDFLoader, silent_errors=True).load())
            except: pass
            try: docs.extend(DirectoryLoader(root, glob="*.docx", loader_cls=Docx2txtLoader, silent_errors=True).load())
            except: pass
            for doc in docs: doc.metadata["category"] = category
            all_documents.extend(docs)

        if not all_documents: return "⚠️ Chưa tìm thấy tài liệu nào."
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)

        if self.api_key:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
            try:
                self.vector_store = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=self.persist_directory)
                return f"✅ THÀNH CÔNG: Đã học xong {len(all_documents)} tài liệu."
            except Exception as e: return f"❌ LỖI INDEXING: {str(e)}"
        else: return "❌ LỖI: Chưa có GOOGLE_API_KEY."

    def retrieve_answer(self, query, chat_history="", category=None):
        """
        Hàm trả lời câu hỏi - PHIÊN BẢN SỬA LỖI INPUT KEYS
        """
        if not self.api_key: return "Lỗi: Chưa cấu hình API Key."
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            google_api_key=self.api_key, 
            temperature=0.3,        
            max_output_tokens=8192,
            timeout=None,
            max_retries=2
        )
        
        # --- FIX LỖI Ở ĐÂY ---
        # 1. Xử lý chat_history bằng Python thuần túy để tránh lỗi ký tự đặc biệt
        safe_history = chat_history.replace("{", "(").replace("}", ")")
        
        # 2. Bơm thẳng history vào string (F-string injection)
        # Lưu ý: Phải dùng {{context}} và {{question}} (2 dấu ngoặc) để giữ lại biến cho LangChain
        template = f"""Bạn là "Trợ lý HR Tận tâm" của Công ty Takagi Việt Nam.
        
        LỊCH SỬ TRÒ CHUYỆN:
        {safe_history}

        DỮ LIỆU TRA CỨU:
        {{context}}

        CÂU HỎI CỦA NHÂN VIÊN: "{{question}}"

        QUY TẮC TRẢ LỜI (ZALO):
        1. KHÔNG DÙNG BẢNG (No Tables). Dùng gạch đầu dòng.
        2. Thân thiện, chính xác số liệu.
        3. Kết hợp lịch sử chat để hiểu câu hỏi cộc lốc.

        PHẢN HỒI:"""
        
        # 3. Khai báo PromptTemplate CHỈ CÒN 2 biến (context, question)
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"], # Đã bỏ chat_history ra khỏi danh sách này
            template=template
        )
        
        search_kwargs = {"k": 6}
        if category: search_kwargs["filter"] = {"category": category}

        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": QA_CHAIN_PROMPT
            }
        )
        
        try:
            # 4. Invoke chỉ cần query (LangChain sẽ tự điền vào {{question}})
            result = qa_chain.invoke({"query": query})
            return result["result"]
        except Exception as e:
            return f"Xin lỗi, hệ thống đang bận. (Lỗi chi tiết: {str(e)})"