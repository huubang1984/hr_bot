import os
import shutil
# Import các loader cho PDF và Word
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
            
        # Sử dụng Model mới nhất mà bạn có quyền truy cập
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=self.api_key, temperature=0.3)
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        return qa_chain.invoke(query)["result"]
