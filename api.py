import os
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.rag_engine import EnterpriseRAG
from dotenv import load_dotenv

# Load biến môi trường từ file .env (nếu chạy local)
load_dotenv()

# --- CẤU HÌNH BẢO MẬT & CHỐNG SPAM ---
# 1. API Secret Token: Mật khẩu để n8n gọi vào.
# Nếu không thiết lập trong Env, mặc định sẽ là "takagi_secret_2025"
API_SECRET_TOKEN = os.getenv("API_SECRET_TOKEN", "takagi_secret_2025")

# 2. Rate Limiting: Giới hạn tốc độ gọi API để tránh quá tải
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Takagi HR Bot Enterprise API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Khởi tạo RAG Engine
rag = EnterpriseRAG()

# --- BỘ NHỚ TẠM (RAM) ---
# Lưu lịch sử chat.
# Cấu trúc: { "zalo_id_123": "User: hi\nBot: hello", ... }
# Lưu ý: Sẽ mất khi Restart server Render (Chấp nhận được với bản Free).
chat_sessions = {}

# Định nghĩa dữ liệu đầu vào từ n8n
class N8NRequest(BaseModel):
    question: str
    session_id: str   # Bắt buộc: ID người dùng (Zalo ID / Telegram ID) để nhớ ngữ cảnh
    api_key: str = None # Tùy chọn: Nếu server chưa set key cứng

# Hàm kiểm tra bảo mật (Dependencies)
async def verify_token(x_token: str = Header(None)):
    """Kiểm tra xem request có gửi kèm header 'x-token' đúng mật khẩu không"""
    if x_token != API_SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="⛔ Unauthorized: Sai mã bảo mật.")

@app.get("/")
def home():
    return {"status": "✅ Takagi HR Bot System is Running", "mode": "Enterprise"}

@app.post("/chat")
@limiter.limit("20/minute") # Giới hạn: 1 người chỉ được hỏi 20 câu/phút
async def chat(
    request: Request, 
    body: N8NRequest, 
    token_check: None = Depends(verify_token) # Kích hoạt kiểm tra bảo mật
):
    try:
        # Cấu hình API Key nếu n8n gửi sang (ghi đè env server)
        if body.api_key:
            rag.api_key = body.api_key
        
        # 1. LẤY TRÍ NHỚ (MEMORY)
        user_id = body.session_id
        # Lấy lịch sử cũ, nếu chưa có thì là chuỗi rỗng
        current_history = chat_sessions.get(user_id, "")
        
        # 2. SUY LUẬN (RAG ENGINE)
        # Gửi câu hỏi + Lịch sử sang cho AI xử lý
        response_text = rag.retrieve_answer(body.question, chat_history=current_history)
        
        # 3. CẬP NHẬT TRÍ NHỚ
        # Ghi thêm câu hỏi mới và câu trả lời mới vào sổ ký ức
        updated_history = current_history + f"\nUser: {body.question}\nBot: {response_text}\n"
        
        # Cắt ngắn lịch sử nếu quá dài (chỉ nhớ khoảng 2000 ký tự cuối để tiết kiệm bộ nhớ AI)
        if len(updated_history) > 2000:
            updated_history = updated_history[-2000:]
        
        # Lưu lại vào RAM
        chat_sessions[user_id] = updated_history
        
        return {
            "status": "success",
            "answer": response_text
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
async def reindex(token_check: None = Depends(verify_token)):
    """API để kích hoạt AI học lại tài liệu từ xa"""
    try:
        result = rag.index_knowledge_base()
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset_memory")
async def reset_memory(body: N8NRequest, token_check: None = Depends(verify_token)):
    """Xóa lịch sử trò chuyện của một người dùng"""
    if body.session_id in chat_sessions:
        del chat_sessions[body.session_id]
    return {"status": "success", "message": "Đã xóa ký ức người dùng này."}