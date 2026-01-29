import os
import shutil
# Import thư viện Google GenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

class EnterpriseRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.vector_store = None
        # Đổi tên biến môi trường cho đúng chuẩn
        self.api_key = os.getenv("GOOGLE_API_KEY")

    def index_knowledge_base(self):
        # 1. Xóa DB cũ đi vì Embeddings của Gemini khác OpenAI
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print("-> Đã xóa DB cũ để tạo mới cho Gemini.")
            except:
                pass

        if not os.path.exists("data"):
            os.makedirs("data")
            return "Thư mục 'data' trống."

        # 2. Load Documents
        try:
            loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            documents = loader.load()
        except Exception as e:
            return f"Lỗi đọc file: {str(e)}"
        
        if not documents: return "Không tìm thấy file .txt."

        # 3. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # 4. Create Vector Store với GEMINI
        if self.api_key:
            # Sử dụng model embedding của Google
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
            try:
                self.vector_store = Chroma.from_documents(
                    documents=texts, 
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
                return f"✅ THÀNH CÔNG (Gemini): Đã index {len(documents)} tài liệu."
            except Exception as e:
                return f"❌ Lỗi Indexing: {str(e)}"
        else:
            return "Cần Google API Key."

    def retrieve_answer(self, query):
        if not self.api_key: return "Vui lòng nhập API Key."
            
        # Setup Embeddings Google
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
            
        # Setup LLM Gemini (Dùng bản Flash cho nhanh và rẻ)
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=self.api_key, temperature=0.3)
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        return qa_chain.invoke(query)["result"]