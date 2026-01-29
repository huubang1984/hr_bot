import sys
import os

print("1. Python Executable đang chạy:", sys.executable)
print("2. Đường dẫn tìm kiếm thư viện (sys.path):")
for p in sys.path:
    print("   -", p)

print("\n3. Thử import langchain...")
try:
    import langchain
    print("   -> OK: Import được langchain từ:", langchain.__file__)
except ImportError as e:
    print("   -> LỖI: ", e)

print("\n4. Thử import langchain.chains...")
try:
    import langchain.chains
    print("   -> OK: Import được chains!")
except ImportError as e:
    print("   -> LỖI: ", e)