# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_engine import EnterpriseRAG
import os
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

app = FastAPI(title="Takagi HR RAG API")

# 1. Khởi tạo bộ não RAG
rag = EnterpriseRAG()

# 2. Định nghĩa dữ liệu đầu vào (từ n8n gửi sang)
class QueryRequest(BaseModel):
    question: str
    api_key: str = None # n8n có thể gửi kèm key hoặc dùng key mặc định của server

# 3. Tạo đường dẫn (Endpoint) để n8n gọi vào
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    try:
        # Nếu n8n gửi key riêng thì dùng, không thì dùng key của server
        if request.api_key:
            rag.api_key = request.api_key
        
        # Gọi hàm retrieve_answer quen thuộc
        response = rag.retrieve_answer(request.question)
        
        return {
            "status": "success",
            "answer": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. API để kích hoạt Re-index từ xa (nếu muốn n8n tự động update dữ liệu)
@app.post("/reindex")
async def reindex_endpoint():
    try:
        result = rag.index_knowledge_base()
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chạy server: uvicorn api:app --reload --port 8000

class N8NRequest(BaseModel):
    question: str
    api_key: str = None
    category: str = None  # Thêm trường này (Ví dụ: "HR", "IT", hoặc để trống)

@app.post("/chat")
async def chat(request: N8NRequest):
    try:
        if request.api_key:
            rag.api_key = request.api_key
        
        # Truyền category vào hàm xử lý
        response = rag.retrieve_answer(request.question, category=request.category)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))