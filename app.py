import streamlit as st
import os
# Import class xử lý RAG từ backend đã sửa
from src.rag_engine import EnterpriseRAG

# 1. Cấu hình trang Web
st.set_page_config(page_title="Trợ lý HR - Takagi VN", page_icon="🤖", layout="wide")

st.title("🤖 Trợ lý ảo HR - Hỏi đáp Chính sách & Quy trình")
st.markdown("---")

# 2. Sidebar (Cột bên trái): Cấu hình & Dữ liệu
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    
    # Nhập API Key (Bắt buộc để chạy)
    user_api_key = st.text_input("Nhập GOOGLE API Key", type="password", placeholder="sk-...")
    
    # Nút nạp lại dữ liệu
    st.subheader("📚 Dữ liệu Quy trình")
    if st.button("🔄 Cập nhật/Re-index Dữ liệu"):
        if not user_api_key:
            st.error("⚠️ Vui lòng nhập API Key trước khi Index.")
        else:
            with st.spinner("Đang đọc tài liệu và huấn luyện AI..."):
                # Khởi tạo engine
                os.environ["GOOGLE_API_KEY"] = user_api_key
                rag = EnterpriseRAG()
                
                # Gọi hàm index
                status = rag.index_knowledge_base()
                
                if "THÀNH CÔNG" in status:
                    st.success(status)
                else:
                    st.error(status)
    
    st.info("💡 Mẹo: Copy file .txt vào thư mục 'data/' sau đó bấm nút trên để AI học kiến thức mới.")

# 3. Giao diện Chat (Phần chính)
# Khởi tạo lịch sử chat nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn! Tôi là trợ lý ảo HR. Bạn cần hỏi về quy trình nghỉ phép, công tác phí hay quy định nào?"}]

# Hiển thị lịch sử chat cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Xử lý khi người dùng nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn ở đây..."):
    # Hiển thị câu hỏi của người dùng
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Xử lý trả lời
    if not user_api_key:
        error_msg = "⚠️ Vui lòng nhập GOOGLE API Key ở cột bên trái để tôi có thể trả lời."
        st.chat_message("assistant").markdown(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.spinner("AI đang tra cứu tài liệu..."):
            try:
                # Thiết lập môi trường
                os.environ["GOOGLE_API_KEY"] = user_api_key
                rag = EnterpriseRAG()
                
                # Gọi hàm trả lời từ backend
                final_answer = rag.retrieve_answer(prompt)
                
                # Hiển thị câu trả lời
                st.chat_message("assistant").markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")