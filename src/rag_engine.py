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
        # BẢO MẬT 1: Lấy Key từ biến môi trường (An toàn hơn hard-code)
        self.api_key = os.getenv("GOOGLE_API_KEY")

   def index_knowledge_base(self):
        # 1. Xóa DB cũ
        if os.path.exists(self.persist_directory):
            try: shutil.rmtree(self.persist_directory)
            except: pass

        if not os.path.exists("data"): return "Thư mục 'data' trống."

        all_documents = []
        print("--- Đang quét và phân loại tài liệu ---")

        # 2. Quét từng thư mục con để gắn thẻ (Metadata)
        # Duyệt qua các folder con trong 'data': HR, IT, Production...
        for category in os.listdir("data"):
            category_path = os.path.join("data", category)
            
            # Chỉ xử lý nếu là thư mục
            if os.path.isdir(category_path):
                print(f"📂 Đang xử lý danh mục: {category}")
                
                docs = []
                # Load TXT
                try: docs.extend(DirectoryLoader(category_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, silent_errors=True).load())
                except: pass
                # Load PDF
                try: docs.extend(DirectoryLoader(category_path, glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True).load())
                except: pass
                # Load Word
                try: docs.extend(DirectoryLoader(category_path, glob="**/*.docx", loader_cls=Docx2txtLoader, silent_errors=True).load())
                except: pass

                # QUAN TRỌNG: Gắn thẻ category cho từng trang tài liệu
                for doc in docs:
                    doc.metadata["category"] = category  # Ví dụ: category = "HR"
                
                all_documents.extend(docs)

        if not all_documents: return "Không tìm thấy tài liệu nào."

        # 3. Chia nhỏ văn bản
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)

        # 4. Tạo Vector Store với Metadata
        if self.api_key:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
            try:
                self.vector_store = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=self.persist_directory)
                return f"✅ Đã học xong {len(all_documents)} tài liệu chia theo các danh mục."
            except Exception as e:
                return f"❌ Lỗi Indexing: {str(e)}"
        return "Thiếu API Key."

   # Thêm tham số category=None (Mặc định là tìm tất cả nếu không chỉ định)
    def retrieve_answer(self, query, category=None):
        if not self.api_key: return "Chưa cấu hình API Key."
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=self.api_key, 
            temperature=0.3,
            max_output_tokens=8192
        )
        
        # --- CẤU HÌNH "TỰ NHIÊN & CÁ NHÂN HÓA" ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            google_api_key=self.api_key, 
            temperature=0.3,        # Tăng nhẹ lên 0.3 để văn phong mềm mại, bớt máy móc (nhưng vẫn chuẩn xác)
            max_output_tokens=8192, # Cho phép trả lời dài đầy đủ
            timeout=None,
            max_retries=2
        )
        
        # --- PROMPT: TẠO NÊN TÍNH CÁCH (PERSONA) ---
        template = """Bạn là "Trợ lý HR Tận tâm" của Takagi Việt Nam. 
        Bạn không phải là cái máy đọc luật, mà là người đồng hành giúp nhân viên giải quyết vấn đề.

        Ngữ cảnh (Thông tin nội bộ):
        {context}

        Câu hỏi của nhân viên: "{question}"

        HƯỚNG DẪN TRẢ LỜI (BẢO MẬT & TỰ NHIÊN):
        1. **Giọng văn:** Thân thiện, lịch sự, dùng từ ngữ "chúng ta", "bạn", "công ty". Tránh dùng từ ngữ quá hành chính cứng nhắc.
        2. **Sự thấu cảm:** Nếu câu hỏi liên quan đến quyền lợi (ốm đau, thai sản, kỷ luật), hãy bắt đầu bằng sự chia sẻ hoặc trấn an (Ví dụ: "Mình rất tiếc nghe tin bạn ốm...", "Về vấn đề này, bạn đừng lo lắng quá...").
        3. **Trình bày:** - Giải thích ngắn gọn trước.
           - Nếu có số liệu/quy trình phức tạp -> Dùng Bảng Markdown hoặc Gạch đầu dòng.
           - Luôn trích dẫn nguồn văn bản (Ví dụ: Theo Điều 5 - Nội quy...).
           - Phải trả lời hết ý, không được dừng giữa chừng.
           - Nếu bảng dữ liệu quá dài, hãy tách thành nhiều bảng nhỏ hoặc dùng danh sách gạch đầu dòng (bullet points) để đảm bảo hiển thị đủ nội dung.
           - Tùy từng nội dung cần thiết, có thể thể hiện bằng đồ họa cho trực quan.           
        4. **Bảo mật:** Chỉ trả lời dựa trên thông tin được cung cấp. Tuyệt đối không bịa đặt hoặc tiết lộ thông tin lương thưởng của người khác nếu không có trong ngữ cảnh.
        5. **Kết thúc:** Luôn đề nghị hỗ trợ thêm (Ví dụ: "Nếu cần mẫu đơn, bạn cứ bảo mình nhé!").

        PHẢN HỒI CỦA BẠN:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
       # --- KỸ THUẬT FILTERING (LỌC) ---
        search_kwargs = {"k": 6}
        
        # Nếu người dùng chỉ định tìm trong HR, chỉ tìm tài liệu có metadata category='HR'
        if category and category != "All":
            search_kwargs["filter"] = {"category": category}
            print(f"🔍 Đang lọc tìm kiếm trong danh mục: {category}")

        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        return qa_chain.invoke(query)["result"]
        
