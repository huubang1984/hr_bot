import os
import shutil
import time

# --- CẤU HÌNH GOOGLE CHAT (REST) ---
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
        
        # HuggingFace API
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
        print("--- 🚀 START INDEXING V4 (ANTI-COLD-START) ---")
        
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

        # 4. Lưu vào DB (Batch nhỏ + Retry mạnh mẽ)
        try:
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
            batch_size = 5  # Giảm xuống 5 để cực kỳ nhẹ
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                current_batch_num = i//batch_size + 1
                
                # CƠ CHẾ THỬ LẠI 5 LẦN (Để chờ Model thức dậy)
                max_retries = 5
                success = False
                
                for attempt in range(max_retries):
                    try:
                        self.vector_store.add_documents(batch)
                        success = True
                        print(f"✅ Đã xong lô {current_batch_num}/{total_batches}")
                        time.sleep(0.5) 
                        break 
                    except Exception as e:
                        # Bắt lỗi KeyError (dấu hiệu model đang ngủ)
                        err_msg = str(e)
                        if "KeyError" in type(e).__name__ or "'0'" in err_msg:
                            print(f"⚠️ Model đang ngủ... Đợi 5s để gọi dậy (Lần {attempt+1})")
                            time.sleep(5) # Ngủ lâu hơn để chờ model load
                        else:
                            print(f"⚠️ Lỗi mạng lô {current_batch_num}: {err_msg}. Thử lại...")
                            time.sleep(2)
                
                if not success:
                    return f"❌ Thất bại tại lô {current_batch_num}. HuggingFace đang quá tải."
                
            return f"✅ (V4) Thành công! Đã học xong {len(all_documents)} tài liệu."
            
        except Exception as e:
            return f"❌ Lỗi Indexing V4: {str(e)}"

    def retrieve_answer(self, query, chat_history="", category=None):
        if not self.api_key: return "Lỗi: Chưa cấu hình Google API Key."
        if not self.embedding_model: return "Lỗi: Chưa cấu hình HuggingFace Token."
            
        self.vector_store = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embedding_model
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=self.api_key, 
            temperature=0.1,
            transport="rest"
        )
        
        search_kwargs = {"k": 5}
        if category: search_kwargs["filter"] = {"category": category}

        try:
            retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            relevant_docs = retriever.invoke(query)
            
            if not relevant_docs:
                return "Hệ thống chưa có dữ liệu. Vui lòng chạy Re-index."
                
        except Exception as e:
            return f"Lỗi truy vấn DB: {str(e)}"
        
        formatted_context = ""
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source_name", "Tài liệu nội bộ")
            content = doc.page_content.replace("\n", " ")
            formatted_context += f"[Nguồn {i+1}: {source}]\nNội dung: {content}\n\n"

        safe_history = chat_history.replace("{", "(").replace("}", ")")
        
        prompt = f"""Bạn là Trợ lý HR của Takagi Việt Nam.
        DỮ LIỆU: {formatted_context}
        LỊCH SỬ: {safe_history}
        CÂU HỎI: "{query}"
        TRẢ LỜI:"""
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Lỗi Gemini: {str(e)}"