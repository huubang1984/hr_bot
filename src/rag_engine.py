mport os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
# Thay đổi quan trọng: Dùng HuggingFace (Local) thay vì Google API cho Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

class EnterpriseRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.vector_store = None
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # SỬ DỤNG LOCAL EMBEDDINGS (Miễn phí, Ổn định)
        # Model 'all-MiniLM-L6-v2' rất nhẹ và hiệu quả
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def index_knowledge_base(self):
        # 1. Dọn dẹp DB cũ
        if os.path.exists(self.persist_directory):
            try: shutil.rmtree(self.persist_directory)
            except: pass

        if not os.path.exists("data"):
            os.makedirs("data")
            return "Folder data created. Please upload files."
            
        all_documents = []
        print("--- 🚀 START INDEXING WITH LOCAL EMBEDDINGS ---")
        
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
                file_name = os.path.basename(doc.metadata.get("source", ""))
                doc.metadata["source_name"] = file_name
            
            all_documents.extend(docs)

        if not all_documents: return "No documents found."
        
        # 3. Cắt nhỏ văn bản
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(all_documents)

        # 4. Lưu vào Vector DB (Dùng Local Embeddings)
        try:
            self.vector_store = Chroma.from_documents(
                documents=texts, 
                embedding=self.embedding_model, # Dùng model nội bộ
                persist_directory=self.persist_directory
            )
            return f"✅ Đã học xong {len(all_documents)} tài liệu bằng Local Embeddings."
        except Exception as e:
            return f"❌ Lỗi Indexing: {str(e)}"

    def retrieve_answer(self, query, chat_history="", category=None):
        if not self.api_key: return "Lỗi: Chưa cấu hình API Key cho Chat."
            
        # Dùng Local Embeddings để tìm kiếm
        self.vector_store = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embedding_model
        )
        
        # Dùng Google Gemini để TRẢ LỜI (Phần này vẫn cần API Key, và nó đang hoạt động tốt)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=self.api_key, 
            temperature=0.1,
            max_output_tokens=8192
        )
        
        # --- TÌM KIẾM DỮ LIỆU ---
        search_kwargs = {"k": 5}
        if category: search_kwargs["filter"] = {"category": category}

        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        relevant_docs = retriever.invoke(query)
        
        # Xây dựng Context
        formatted_context = ""
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source_name", "Tài liệu nội bộ")
            content = doc.page_content.replace("\n", " ")
            formatted_context += f"[Nguồn {i+1}: {source}]\nNội dung: {content}\n\n"

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
