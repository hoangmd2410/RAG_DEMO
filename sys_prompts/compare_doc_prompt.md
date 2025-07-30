**Truy vấn tìm kiếm:**
{query}

**Các điều khoản được trích xuất:**
{context_text}

**Yêu cầu phân tích:**
Dựa trên các điều khoản được cung cấp, vui lòng thực hiện các bước sau:

1. Xác định bất kỳ điều khoản nào có nội dung **mâu thuẫn trực tiếp** với nhau.
2. Xác định các điều khoản có nội dung **tương tự hoặc trùng lặp** với nhau một cách đáng kể.
3. Với mỗi phát hiện, cung cấp lời giải thích ngắn gọn và trích dẫn rõ ràng các tài liệu và điều khoản liên quan.

Phản hồi theo định dạng JSON có cấu trúc như sau:
{{
            "type": "Conflict" | "Similarity",
            "explanation": "Giải thích lý do tại sao các điều khoản này mâu thuẫn hoặc giống nhau",
            "clause": "nội dung tóm tắt điều khoản mã số hiệu mà giống nhau hoặc mâu thuẫn"
}}


Nếu không tìm thấy bất kỳ mâu thuẫn hoặc điểm tương đồng nào, vui lòng trả về danh sách rỗng:
{{}}