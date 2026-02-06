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
        # 1. Dọn dẹp & Chuẩn bị
        if os.path.exists(self.persist_directory):
            try: shutil.rmtree(self.persist_directory)
            except: pass
        if not os.path.exists("data"):
            os.makedirs("data")
            return "Folder data created. Please upload files."
            
        all_documents = []
        print("--- 🚀 START INDEXING ---")
        
        # 2. Quét tài liệu & Gắn metadata
        for root, dirs, files in os.walk("data"):
            category = os.path.basename(root) if root != "data" else "General"
            docs = []
            try: docs.extend(DirectoryLoader(root, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, silent_errors=True).load())
            except: pass
            try: docs.extend(DirectoryLoader(root, glob="*.pdf", loader_cls=PyPDFLoader, silent_errors=True).load())
            except: pass
            try: docs.extend(DirectoryLoader(root, glob="*.docx", loader_cls=Docx2txtLoader, silent_errors=True).load())
            except: pass
            
            # Gắn tên file vào metadata để AI biết nguồn
            for doc in docs: 
                doc.metadata["category"] = category
                # Lưu tên file gốc (ví dụ: Noi_quy_2025.pdf)
                doc.metadata["source"] = os.path.basename(doc.metadata.get("source", ""))
            
            all_documents.extend(docs)

        if not all_documents: return "No documents found to index."
        
        # 3. Cắt nhỏ (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)

        # 4. Lưu vào Vector DB
        if self.api_key:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
            self.vector_store = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=self.persist_directory)
            return f"✅ Indexed {len(all_documents)} files ({len(texts)} chunks)."
        return "Missing API Key."

    def retrieve_answer(self, query, chat_history="", category=None):
        if not self.api_key: return "Lỗi: Chưa cấu hình API Key."
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        
        # Model Flash cho tốc độ nhanh và ổn định
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=self.api_key, 
            temperature=0.2, 
            max_output_tokens=8192,
timeout=None,
max_retries=2
        )
        
        # --- KỸ THUẬT NHỒI NGUỒN (CONTEXT INJECTION) ---
        # Thay vì để LangChain tự làm, ta tự tìm kiếm và format dữ liệu đầu vào
        search_kwargs = {"k": 5}
        if category: search_kwargs["filter"] = {"category": category}
        
        # 1. Tìm 5 đoạn văn bản liên quan nhất
        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(query)
        
        # 2. Ghép nội dung + Tên nguồn vào Context
        context_text = ""
        for doc in docs:
            source_name = doc.metadata.get("source", "Tài liệu nội bộ")
            content = doc.page_content.replace("\n", " ")
            context_text += f"- Trích từ tài liệu [{source_name}]: {content}\n\n"

        # 3. Xử lý lịch sử chat
        safe_history = chat_history.replace("{", "(").replace("}", ")")
        
        # --- PROMPT KỶ LUẬT THÉP ---
        prompt = f"""Bạn là Trợ lý HR chuyên nghiệp và tận tâm của Takagi Việt Nam.
        
        NHIỆM VỤ: Trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.
        
        QUY TẮC BẮT BUỘC (TUÂN THỦ TUYỆT ĐỐI):
        1. **CHỈ SỬ DỤNG** thông tin trong phần "DỮ LIỆU TRA CỨU" bên dưới.
        2. **KHÔNG** được tự bịa ra kiến thức bên ngoài (nếu không có trong tài liệu, hãy nói: "Xin lỗi, tôi chưa tìm thấy thông tin này trong tài liệu nội bộ").
        3. **TRÍCH DẪN NGUỒN:** Cuối mỗi ý hoặc cuối câu trả lời, PHẢI ghi rõ thông tin lấy từ đâu.
           - Ví dụ: "...theo quy định mới (Nguồn: Noi_quy_2025.pdf)".
        4. Trình bày gạch đầu dòng, ngắn gọn, dễ đọc trên điện thoại.

QUY TẮC TRẢ LỜI (ZALO):
1. KHÔNG DÙNG BẢNG (No Tables). Dùng gạch đầu dòng.
2. Thân thiện, chính xác số liệu.
3. Kết hợp lịch sử chat để hiểu câu hỏi cộc lốc.

        ---
        LỊCH SỬ CHAT:
        {safe_history}
        ---
        DỮ LIỆU TRA CỨU (CONTEXT):
        {context_text}
        ---
        CÂU HỎI CỦA NHÂN VIÊN: "{query}"
        
        TRẢ LỜI:"""
        
        # Gọi thẳng LLM (Bỏ qua Chain phức tạp để kiểm soát tốt hơn)
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Lỗi xử lý: {str(e)}"
