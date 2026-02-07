import os
import shutil
import time

# --- CẤU HÌNH GOOGLE CHAT ---
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
import google.generativeai as genai
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"), transport="rest")

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

class EnterpriseRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.vector_store = None
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if self.hf_token:
            self.embedding_model = HuggingFaceInferenceAPIEmbeddings(
                api_key=self.hf_token,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embedding_model = None

    def index_knowledge_base(self):
        if not self.hf_token: return "❌ Lỗi: Thiếu HUGGINGFACEHUB_API_TOKEN."

        # 1. Dọn dẹp DB cũ
        if os.path.exists(self.persist_directory):
            try: shutil.rmtree(self.persist_directory)
            except: pass

        if not os.path.exists("data"):
            os.makedirs("data")
            return "Folder data created."
            
        all_documents = []
        print("--- 🚀 START INDEXING WITH ROBUST RETRY ---")
        
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

        # 4. Lưu vào DB (Batch nhỏ + Retry)
        try:
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
            # --- CẤU HÌNH AN TOÀN ---
            batch_size = 10  # Giảm xuống 10 để cực kỳ an toàn
            # ------------------------
            
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                current_batch_num = i//batch_size + 1
                
                # CƠ CHẾ THỬ LẠI (RETRY) KHI MẤT MẠNG
                max_retries = 3
                success = False
                
                for attempt in range(max_retries):
                    try:
                        self.vector_store.add_documents(batch)
                        success = True
                        print(f"✅ Đã xong lô {current_batch_num}/{total_batches}")
                        time.sleep(1) # Nghỉ 1s
                        break # Thành công thì thoát vòng lặp thử lại
                    except Exception as e:
                        print(f"⚠️ Lỗi lô {current_batch_num} (Lần thử {attempt+1}): {str(e)}")
                        time.sleep(3) # Nghỉ 3s rồi thử lại
                
                if not success:
                    return f"❌ Thất bại tại lô {current_batch_num} sau 3 lần thử. Vui lòng kiểm tra lại mạng."
                
            return f"✅ Thành công! Đã học xong {len(all_documents)} tài liệu ({len(texts)} đoạn)."
            
        except Exception as e:
            return f"❌ Lỗi Indexing Fatal: {str(e)}"

    def retrieve_answer(self, query, chat_history="", category=None):
        if not self.api_key: return "Lỗi: Chưa cấu hình Google API Key."
        if not self.embedding_model: return "Lỗi: Chưa cấu hình HuggingFace Token."
            
        # Kết nối DB
        self.vector_store = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embedding_model
        )
        
        # Model Chat (Google Gemini qua REST)
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
            
            if not relevant_docs:
                return "Hệ thống chưa có dữ liệu."
                
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
        1. Trả lời dựa trên dữ liệu tra cứu.
        2. Nếu không có thông tin, nói "Xin lỗi, không tìm thấy trong tài liệu".
        3. Ghi nguồn ở cuối câu trả lời.
        
        TRẢ LỜI:"""
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Lỗi Gemini: {str(e)}"