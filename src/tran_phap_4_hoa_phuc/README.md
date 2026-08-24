# Trận Pháp 4: Bàn Cờ Nhân Quả - Họa Phúc Song Hành (Quantifying Luck)

> _"Trong họa có phúc, trong phúc có họa. Cả hai đều là... label của con người."_

## Ý tưởng

Xây một quần thể giả lập với 2 biến cố nhị phân:
- **A = Trúng Jackpot**
- **B = Gặp Họa** (tai nạn/biến cố xui xảy ra sau đó)

**A và B được sinh ĐỘC LẬP TUYỆT ĐỐI theo thiết kế mặc định** - đúng giả thuyết mà bí kíp muốn kiểm chứng: "may mắn không phải một loại linh khí trong vũ trụ, chỉ là cái nhãn con người tự dán lên các sự kiện ngẫu nhiên". 

Từ quần thể đó, trận pháp:

1. Dựng **Bàn Cờ Nhân Quả** - ma trận 2x2 đếm số người theo từng tổ hợp (Trúng+Họa, Trúng+Không, Không Trúng+Họa, Không Trúng+Không)
2. Kiểm định thống kê **Chi-square test of independence** - nếu A, B thực sự độc lập, p-value phần lớn phải > 0.05 (không đủ bằng chứng bác bỏ độc lập)
3. So sánh trực tiếp **P(Họa | Trúng)** vs **P(Họa | Không Trúng)** kèm khoảng tin cậy 95% - khoảng tin cậy chồng lấn nhau chứng minh không có "lời nguyền" nào phân biệt được khỏi nhiễu ngẫu nhiên.
4. Chạy **200 lần mô phỏng độc lập** để dựng "phân phối nhiễu do may rủi" - cho thấy một câu chuyện đơn lẻ (như 1 bài báo về "lời nguyền xổ số") hoàn toàn có thể chỉ là 1 điểm ngẫu nhiên trong phân phối này, không phải bằng chứng của quan hệ nhân quả thật.

## Chân lý hé lộ (chạy thật, 2 triệu người, 1994 người "trúng số")

```
P(Họa | Trúng)        = 4.9649%
P(Họa | Không Trúng)  = 5.0203%
Chênh lệch             = -0.0554 điểm %
Odds Ratio             = 0.993   (~1.0 = không liên hệ)
Chi² p-value           = 0.9506  (KHÔNG có ý nghĩa thống kê)
```

Qua 200 lần chạy độc lập, chênh lệch dao động ngẫu nhiên quanh **+0.12 điểm %** (độ lệch chuẩn 1.6) - một kết quả đơn lẻ dương hay âm vài điểm % hoàn toàn nằm trong biên độ nhiễu bình thường.

## Chạy thử

```bash
pip install -r requirements.txt

python -m src.tran_phap_4_hoa_phuc.run
```

Tinh chỉnh:
```bash
# Xác suất "trúng số" trong mô phỏng - mặc định phóng đại lên (0.1%) để có
# đủ mẫu người trúng phân tích thống kê (xổ số thật ~1/29 triệu, xem Trận Pháp 1)
python -m src.tran_phap_4_hoa_phuc.run --p-win 0.0005

# Xác suất nền "gặp họa" (số minh họa, không phải số liệu dịch tễ thật)
python -m src.tran_phap_4_hoa_phuc.run --p-hoa-baseline 0.03

# Thêm vào để tự nghịch: import trực tiếp và đặt true_effect != 0 để xem
# nếu THẬT SỰ có một "lời nguyền" thì Bàn Cờ Nhân Quả trông khác thế nào
python -c "
from src.tran_phap_4_hoa_phuc import simulate_population, build_contingency_table, analyze_independence
pop = simulate_population(1_000_000, p_win=0.001, p_hoa_baseline=0.05, true_effect=0.3, seed=1)
table = build_contingency_table(pop['a'], pop['b'])
print(analyze_independence(table))
"
```

Kết quả lưu tại `outputs/tran_phap_4/`:
- `ban_co_nhan_qua.png` - 3 biểu đồ: heatmap Bàn Cờ Nhân Quả, so sánh P(Họa|Trúng) vs P(Họa|Không Trúng), phân phối nhiễu qua 200 lần chạy
- `tong_ket.json` - số liệu đầy đủ

Với quy mô mặc định (2 triệu người, 200 lần chạy null), toàn bộ pipeline chạy trong khoảng **0.6 giây** trên máy CPU 8 luồng ~2.5GHz / 8GB RAM.

## Vì sao không dùng dataset "Lottery Curse" thật từ Mỹ/Âu ?

Bí kíp gốc ("Kiếp Nạn 2") có đề cập mượn dataset công khai về tỷ lệ phá sản/ly hôn/tai nạn của người trúng Powerball.
Tuy nhiên, môi trường chạy code này không có quyền truy cập ổn định vào các nguồn dữ liệu nghiên cứu như vậy, và quan trọng hơn: **dùng số liệu thật mà không kiểm chứng được nguồn gốc/độ tin cậy sẽ nguy hiểm hơn là hữu ích** - dễ vô tình lan truyền số liệu sai lệch dưới vỏ bọc "khoa học".
Vì đây là "Pet Project giải trí trí tuệ", module này chọn hướng minh bạch hơn: **mô phỏng có kiểm soát** với giả thuyết rõ ràng (A, B độc lập), rồi để chính con số tự nói lên bài học - thay vì mượn danh một dataset không kiểm chứng được.
Ai muốn nghiêm túc hơn có thể tự thay `population_model.py` bằng dữ liệu thật đã được thẩm định.

## Kỹ thuật đứng sau

1. **Chi-square test of independence** (`scipy.stats.chi2_contingency`) - công cụ thống kê chuẩn để kiểm định 2 biến nhị phân có độc lập hay không, dùng đúng cách thay vì chỉ nhìn vào con số chênh lệch % rồi "cảm tính" kết luận.
2. **Odds Ratio với hiệu chỉnh liên tục (+0.5)** - tránh chia cho 0 khi một ô trong bàn cờ có count = 0 (dễ xảy ra khi p_win rất nhỏ)
3. **Khoảng tin cậy 95% (xấp xỉ chuẩn)** cho từng tỷ lệ P(Họa|A) - trực quan hóa độ bất định, không chỉ đưa ra 1 con số trần trụi.
4. **"Phân phối nhiễu do may rủi"** qua nhiều trial độc lập (tái sử dụng đúng triết lý "repeated trials" của Trận Pháp 1) - đây là cách chặt chẽ nhất để trả lời câu hỏi "kết quả 1 lần chạy này có bất thường không ?", thay vì chỉ nhìn 1 con số và đoán.
5. **Tham số `true_effect`** trong `population_model.py` cho phép người dùng CHỦ ĐỘNG bật một "lời nguyền" giả định để xem Bàn Cờ Nhân Quả và kiểm định thống kê phản ứng ra sao - dùng để đối chứng, hiểu rõ hơn sự khác biệt giữa "có hiệu ứng thật" và "chỉ là nhiễu ngẫu nhiên".
