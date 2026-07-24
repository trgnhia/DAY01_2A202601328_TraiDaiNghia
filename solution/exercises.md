# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Khi temperature thấp như 0.0, phản hồi ổn định, trực tiếp và ít sáng tạo; cùng một prompt thường cho câu trả lời khá giống nhau. Khi tăng lên 0.7 và 1.2, câu trả lời đa dạng hơn, có nhiều cách diễn đạt và chi tiết thú vị hơn. Ở khoảng 1.8, phản hồi bắt đầu dễ lan man hoặc kém mạch lạc hơn vì model lấy mẫu quá ngẫu nhiên.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Với trợ lý soạn hợp đồng pháp lý, tôi sẽ đặt temperature thấp, khoảng 0.0–0.2, vì cần chính xác, nhất quán và tránh suy diễn sáng tạo. Với trợ lý viết slogan quảng cáo, tôi sẽ đặt cao hơn, khoảng 0.9–1.2, vì nhiệm vụ cần nhiều ý tưởng mới, cách chơi chữ và biến thể sáng tạo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Workload có 20.000 × 2 × 500 = 20.000.000 output token mỗi ngày. Theo bảng giá, `gpt-4o` tốn khoảng 20.000.000 / 1000 × 0.010 = 200 USD/ngày, còn `gpt-4o-mini` tốn khoảng 20.000.000 / 1000 × 0.0006 = 12 USD/ngày. Model lớn xứng đáng khi xử lý việc khó như phân tích pháp lý, y tế hoặc lập luận phức tạp; model nhỏ phù hợp cho tác vụ đơn giản, lặp lại nhiều như tóm tắt ngắn, phân loại intent hoặc trả lời FAQ.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona nhà thơ, phản hồi mềm hơn, dùng hình ảnh ví von và tránh thuật ngữ kỹ thuật, nên dễ hiểu với người mới nhưng ít chính xác theo kiểu kỹ thuật. Với persona kỹ sư senior, phản hồi có cấu trúc hơn, dùng khái niệm như dữ liệu, mô hình, huấn luyện và có thể đưa ví dụ code. Điều này cho thấy system prompt có thể điều khiển giọng văn, mức độ chuyên môn, độ dài, cấu trúc câu trả lời và loại ví dụ được dùng. Nó không chỉ nói “trả lời gì”, mà còn định hình “trả lời như ai” và “trả lời cho đối tượng nào”.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Với đoạn tiếng Việt khoảng 150 từ, ước lượng thô `số từ / 0.75` cho khoảng 200 token, còn `tiktoken` thường có thể cho số cao hơn do tiếng Việt có dấu và nhiều từ bị tách thành nhiều token con. Nếu ví dụ `tiktoken` đếm khoảng 260 token thì chênh lệch là khoảng 30% so với ước lượng thô. Vì vậy khi dự toán ngân sách cho ứng dụng tiếng Việt, dùng công thức thô có nguy cơ dự toán thiếu, đặc biệt với văn bản nhiều dấu, tên riêng hoặc ký hiệu.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản và trợ lý giọng nói hưởng lợi nhiều từ streaming vì người dùng thấy hoặc nghe phản hồi ngay khi model bắt đầu sinh token, làm cảm giác chờ ngắn hơn. Trợ lý giọng nói đặc biệt cần streaming để có thể bắt đầu đọc sớm thay vì đợi toàn bộ câu trả lời hoàn tất. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm không cần streaming nhiều, vì người dùng không theo dõi từng token; thứ quan trọng hơn là độ chính xác, khả năng retry, logging và kết quả cuối cùng.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Exponential backoff làm khoảng chờ tăng dần sau mỗi lần lỗi, nên giảm áp lực lên API khi hệ thống đang quá tải. Nếu dùng delay cố định, hàng nghìn client có thể tiếp tục retry theo cùng nhịp và làm tình trạng quá tải kéo dài. Jitter thêm một lượng trễ ngẫu nhiên để các client không retry đồng loạt cùng thời điểm, giảm hiện tượng “retry storm” và phân tán tải đều hơn.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt tôi dùng: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.” Nếu xóa cụm “trợ giảng thân thiện của khóa AI”, trợ lý có thể trả lời chung chung hơn, ít đặt mình vào bối cảnh lớp học và ít giải thích theo hướng sư phạm. Nếu xóa cụm “trả lời ngắn gọn bằng tiếng Việt”, trợ lý có thể trả lời dài hơn hoặc chuyển sang tiếng Anh khi gặp thuật ngữ kỹ thuật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Ví dụ người dùng ở lượt đầu nói “tôi đang làm app đặt lịch khám, ưu tiên bảo mật dữ liệu bệnh nhân”, sau đó hỏi qua nhiều lượt về UI, API và database; đến lượt thứ 6 họ hỏi “thiết kế schema cho yêu cầu ban đầu”, trợ lý có thể quên bối cảnh y tế và yêu cầu bảo mật vì thông tin đó đã bị cắt khỏi history. Cách khắc phục là lưu một bản tóm tắt dài hạn của các yêu cầu quan trọng, ví dụ domain, ràng buộc bảo mật, quyết định kỹ thuật đã chốt, rồi gửi summary đó kèm 4 lượt gần nhất. Cách này giữ context quan trọng mà không cần gửi toàn bộ lịch sử.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
