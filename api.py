import os
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.rag_engine import EnterpriseRAG
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv()

# --- CẤU HÌNH ---
API_SECRET_TOKEN = os.getenv("API_SECRET_TOKEN", "takagi_secret_2025")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Takagi HR Bot Enterprise API (Docker)")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Khởi tạo RAG Engine
rag = EnterpriseRAG()

# Bộ nhớ tạm
chat_sessions = {}

class N8NRequest(BaseModel):
    question: str
    session_id: str
    api_key: str = None

# Hàm kiểm tra bảo mật
async def verify_token(x_token: str = Header(None)):
    if x_token != API_SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="⛔ Unauthorized: Sai mã bảo mật.")

@app.get("/")
def home():
    return {"status": "✅ System Operational", "mode": "Docker Optimized"}

# --- API CHAT ---
@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, body: N8NRequest, token_check: None = Depends(verify_token)):
    try:
        if body.api_key:
            rag.api_key = body.api_key
        
        user_id = body.session_id
        current_history = chat_sessions.get(user_id, "")
        
        response_text = rag.retrieve_answer(body.question, chat_history=current_history)
        
        updated_history = current_history + f"\nUser: {body.question}\nBot: {response_text}\n"
        if len(updated_history) > 2000: updated_history = updated_history[-2000:]
        chat_sessions[user_id] = updated_history
        
        return {"status": "success", "answer": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- API RE-INDEX (Cái bạn đang thiếu) ---
@app.post("/reindex")
async def reindex(request: Request, x_token: str = Header(None)):
    # Kiểm tra token thủ công cho chắc chắn
    if x_token != API_SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    try:
        # Gọi hàm index từ RAG Engine
        result = rag.index_knowledge_base()
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_memory")
async def reset_memory(body: N8NRequest, token_check: None = Depends(verify_token)):
    if body.session_id in chat_sessions:
        del chat_sessions[body.session_id]
    return {"status": "success", "message": "Memory cleared."}
