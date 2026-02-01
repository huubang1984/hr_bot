from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_engine import EnterpriseRAG
import os
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo API
app = FastAPI()
rag = EnterpriseRAG()

# Định dạng dữ liệu n8n gửi sang
class N8NRequest(BaseModel):
    question: str
    api_key: str = None

@app.get("/")
def home():
    return {"status": "HR Bot API is running"}

@app.post("/chat")
async def chat(request: N8NRequest):
    try:
        # Ưu tiên dùng Key n8n gửi sang, nếu không có thì dùng Key mặc định server
        if request.api_key:
            rag.api_key = request.api_key

        response = rag.retrieve_answer(request.question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
