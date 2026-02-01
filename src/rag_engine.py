import os
import shutil
# Import các loader cho PDF và Word
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
        # 1. Xóa DB cũ để nạp mới sạch sẽ
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
            except:
                pass

        if not os.path.exists("data"):
            os.makedirs("data")
            return "Thư mục 'data' trống."

        documents = []
        print("--- Đang quét tài liệu ---")

        # --- GIAI ĐOẠN 1: ĐỌC FILE TXT ---
        try:
            txt_loader = DirectoryLoader(
                './data', glob="**/*.txt", loader_cls=TextLoader, 
                loader_kwargs={'encoding': 'utf-8'}, silent_errors=True
            )
            documents.extend(txt_loader.load())
        except Exception as e:
            print(f"Lỗi đọc TXT: {e}")

        # --- GIAI ĐOẠN 2: ĐỌC FILE PDF ---
        try:
            pdf_loader = DirectoryLoader(
                './data', glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True
            )
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Lỗi đọc PDF: {e}")

        # --- GIAI ĐOẠN 3: ĐỌC FILE WORD (.DOCX) ---
        try:
            word_loader = DirectoryLoader(
                './data', glob="**/*.docx", loader_cls=Docx2txtLoader, silent_errors=True
            )
            documents.extend(word_loader.load())
        except Exception as e:
            print(f"Lỗi đọc DOCX: {e}")
        
        if not documents: 
            return "Không tìm thấy tài liệu nào (TXT, PDF, DOCX) trong thư mục data."

        print(f"-> Tổng cộng đã tìm thấy {len(documents)} trang tài liệu.")

        # 2. Chia nhỏ văn bản (Split)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 3. Tạo Vector Store
        if self.api_key:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
            try:
                self.vector_store = Chroma.from_documents(
                    documents=texts, 
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                return f"✅ THÀNH CÔNG: Đã học xong {len(documents)} tài liệu (TXT/PDF/DOCX)."
            except Exception as e:
                return f"❌ Lỗi Indexing: {str(e)}"
        else:
            return "Cần Google API Key."

    def retrieve_answer(self, query):
        if not self.api_key: return "Vui lòng nhập API Key."
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        
        # Dùng model Gemini 2.5 Flash như bạn đã cấu hình
        llm = ChatGoogleGenerativeAI(
                model="gemini-3.0 pro ", 
                google_api_key=self.api_key, 
                temperature=0.1,
                max_output_tokens=8192
            )
        
        # --- ĐÂY LÀ PHẦN SUPER PROMPT MỚI ---
        template = """Bạn là Chuyên gia Phân tích Dữ liệu HR (Data Analyst) tại Takagi Việt Nam.
        
        Ngữ cảnh (Context):
        {context}

        Câu hỏi: "{question}"

        YÊU CẦU QUAN TRỌNG:
        1. **Ưu tiên sự trọn vẹn:** Phải trả lời hết ý, không được dừng giữa chừng.
        2. **Xử lý bảng dài:** Nếu bảng dữ liệu quá dài, hãy tách thành nhiều bảng nhỏ hoặc dùng danh sách gạch đầu dòng (bullet points) để đảm bảo hiển thị đủ nội dung.
        3. **Nội dung:** Phân tích sâu, trích dẫn số liệu chính xác (mức phạt, thời hạn, điều kiện).
        4. **Nguồn:** Ghi rõ trích từ văn bản nào (Điều khoản số mấy).
        5. **Giọng văn:** Chuyên nghiệp nhưng gần gũi, khách quan, giống như một bản báo cáo phân tích.
        6. Tùy từng nội dung cần thiết, có thể thể hiện bằng đồ họa cho trực quan.

        TRẢ LỜI CỦA BẠN:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # Nhớ giữ k=10 để có đủ dữ liệu phân tích
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        return qa_chain.invoke(query)["result"]
