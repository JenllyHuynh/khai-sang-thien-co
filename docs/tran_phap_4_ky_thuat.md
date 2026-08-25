# Tài Liệu Kỹ Thuật - Trận Pháp 4: Bàn Cờ Nhân Quả (Quantifying Luck)

## 1. Bài toán cần giải

Kiểm chứng bằng thống kê giả thuyết: **"May mắn không phải một loại linh khí trong vũ trụ - nó chỉ là cái nhãn con người dán lên các sự kiện ngẫu nhiên độc lập."** Cụ thể: liệu "trúng số" (A) và "gặp họa sau đó" (B) - 2 biến cố mà dân gian hay gán ghép thành "lời nguyền xổ số" - có thực sự liên hệ thống kê với nhau, hay chỉ là ngụy biện tường thuật (narrative fallacy) từ việc con người chỉ nhớ những câu chuyện kịch tính?

## 2. Kiến trúc pipeline

```
population_model.py  ->  sinh quần thể với A, B (mặc định ĐỘC LẬP)
contingency.py        ->  ma trận 2×2 + kiểm định Chi-square + odds ratio
narrative_bias.py      ->  chạy nhiều trial độc lập -> phân phối "nhiễu do may rủi"
simulate.py             ->  điều phối
stats.py                ->  tổng hợp
visualize.py             ->  3 biểu đồ: heatmap, so sánh có CI, phân phối null
run.py                   ->  CLI
```

## 3. Mô hình quần thể (`population_model.py`)

### 3.1. Thiết kế biến cố

```python
A ~ Bernoulli(p_win)                                     # "trúng jackpot"
p_hoa_per_person = p_hoa_baseline + true_effect  nếu A=1
                  = p_hoa_baseline                nếu A=0
B ~ Bernoulli(p_hoa_per_person)
```

**Mặc định `true_effect = 0`** -> `B` được sinh với CÙNG xác suất nền `p_hoa_baseline` bất kể `A` là gì -> **A và B độc lập tuyệt đối theo xây dựng (by construction)**. Đây là giả thuyết H0 (null hypothesis) mà toàn bộ trận pháp kiểm chứng.

Tham số `true_effect` được giữ lại (không xóa) để người dùng có thể chủ động BẬT một "lời nguyền" giả định (`true_effect > 0`) nhằm đối chứng - xem Bàn Cờ Nhân Quả và kiểm định phản ứng ra sao KHI THẬT SỰ có quan hệ nhân quả, từ đó hiểu rõ hơn sự khác biệt giữa "có hiệu ứng thật" và "chỉ là nhiễu ngẫu nhiên".

### 3.2. Lựa chọn tham số

- `p_win = 0.001` (0.1%) - **phóng đại so với xác suất trúng jackpot thật** (~3.45×10^-8, xem Trận Pháp 1) để có đủ mẫu người "trúng số" phân tích thống kê trong một quần thể mô phỏng cỡ vài triệu người. Nếu dùng đúng xác suất thật, cần hàng chục triệu người mới có vài người trúng - không đủ mẫu để kiểm định có ý nghĩa. Đây là giả định MINH BẠCH, không phải mô tả xác suất trúng số thật.
- `p_hoa_baseline = 0.05` (5%) - số minh họa cho mục đích giáo dục, không phải số liệu dịch tễ/tài chính đã kiểm chứng.

## 4. Bàn Cờ Nhân Quả - Ma trận 2×2 (`contingency.py`)

### 4.1. Cấu trúc

```
                    B = 1 (Họa)      B = 0 (Không họa)
A = 1 (Trúng)      n(A1,B1)          n(A1,B0)
A = 0 (Không)      n(A0,B1)          n(A0,B0)
```

Tính trực tiếp bằng phép AND/NOT vector hóa trên mảng boolean:

```python
n_a1_b1 = sum(a & b)
n_a1_b0 = sum(a & ~b)
n_a0_b1 = sum(~a & b)
n_a0_b0 = sum(~a & ~b)
```

### 4.2. Kiểm định Chi-square test of independence

**Giả thuyết:**
- H0: A và B độc lập (P(A∩B) = P(A)×P(B) với mọi tổ hợp)
- H1: A và B không độc lập

**Thống kê kiểm định:**

```
χ^2 = Σ (O_ij - E_ij)^2 / E_ij
```

trong đó `O_ij` là số quan sát thực tế ở ô `(i,j)`, `E_ij` là số kỳ vọng NẾU H0 đúng: `E_ij = (tổng hàng i × tổng cột j) / tổng toàn bảng`. 

Code dùng `scipy.stats.chi2_contingency(table, correction=True)` - tham số
`correction=True` áp dụng hiệu chỉnh liên tục Yates (Yates' continuity correction), phù hợp cho bảng 2×2, giảm sai số khi cỡ mẫu ở một số ô không quá lớn.

**Diễn giải `p_value`:** xác suất quan sát được một χ^2 cực đoan như vậy (hoặc hơn) NẾU giả thuyết độc lập (H0) là đúng. Quy ước: `p_value < 0.05` -> đủ căn cứ bác bỏ H0 (có bằng chứng thống kê cho một liên hệ). Vì trận pháp sinh A, B độc lập theo xây dựng, **kỳ vọng phần lớn các lần chạy sẽ cho `p_value > 0.05`** (không bác bỏ được H0) - đúng như thiết kế.

### 4.3. Odds Ratio và Relative Risk

```python
odds_ratio = [(n_a1_b1 + 0.5) × (n_a0_b0 + 0.5)] / [(n_a1_b0 + 0.5) × (n_a0_b1 + 0.5)]
relative_risk = P(B|A=1) / P(B|A=0)
```

Hệ số `+0.5` (Haldane-Anscombe continuity correction) tránh chia cho 0 khi một ô trong bảng có count = 0 - dễ xảy ra khi `p_win` rất nhỏ (số người "trúng và gặp họa" có thể bằng 0 trong một số lần chạy). `odds_ratio ~ 1.0` và `relative_risk ~ 1.0` đều là dấu hiệu của "không có liên hệ".

## 5. So sánh có khoảng tin cậy (`visualize.py`)

### 5.1. Ước lượng khoảng tin cậy 95% (xấp xỉ chuẩn)

Với mỗi tỷ lệ `p = P(B|A)` ước lượng từ `n` mẫu, sai số chuẩn (standard error) theo phân phối nhị thức:

```
SE = √(p(1-p) / n)
CI_95% = p ± 1.96 × SE
```

(Xấp xỉ chuẩn - hợp lệ khi `n×p` và `n×(1-p)` đều đủ lớn, đúng với quy mô mô phỏng của trận pháp.)

### 5.2. Ý nghĩa trực quan

Nếu 2 khoảng tin cậy (của nhóm "trúng số" và nhóm "không trúng") **chồng lấn nhau** trên biểu đồ, đó là bằng chứng trực quan rằng KHÔNG có sự khác biệt đủ tin cậy giữa 2 nhóm - nhất quán với kết quả Chi-square test.

Lưu ý nhóm "trúng số" luôn có `n` nhỏ hơn nhiều (vì `p_win` nhỏ), nên khoảng tin cậy của nhóm này RỘNG HƠN đáng kể - một điểm quan trọng về mặt thống kê: **mẫu nhỏ tự nhiên có độ bất định cao hơn**, một phần lý do khiến các nghiên cứu "lời nguyền xổ số" thật ngoài đời (vốn có rất ít người trúng jackpot để nghiên cứu) khó đưa ra kết luận chắc chắn.

## 6. Ngụy biện tường thuật - Phân phối "nhiễu do may rủi" (`narrative_bias.py`)

### 6.1. Động lực

Một kiểm định thống kê (`p_value`) từ MỘT lần chạy vẫn có thể gây hiểu lầm nếu người đọc không có cảm nhận về "biên độ dao động tự nhiên" của con số đó. Cụ thể: một phóng viên chỉ quan sát ĐÚNG 1 "câu chuyện" (1 lần chạy) rất dễ nhầm một điểm nhiễu ngẫu nhiên thành bằng chứng của "lời nguyền". Nếu tình cờ chênh lệch quan sát được nghiêng về phía có vẻ "kịch tính".

### 6.2. Thiết kế: Repeated Trials (tái dùng triết lý Trận Pháp 1)

```python
for i in range(n_trials):
    pop = simulate_population(..., true_effect=0.0, seed=seed_start + i)   # LUÔN độc lập
    diffs[i] = P(B|A) - P(B|not A)    # tính trên MỖI lần chạy độc lập
```

Với `n_trials = 200` lần chạy độc lập (mỗi lần seed khác nhau, quần thể nhỏ hơn quần thể chính để tiết kiệm chi phí tính toán - `n_people // 10`, tối thiểu 50.000), thu được một MẢNG 200 giá trị chênh lệch. Vì A, B LUÔN được sinh độc lập trong mọi lần chạy này, **phân phối của 200 giá trị này chính là "phân phối null"** (null distribution) - biên độ dao động thuần túy do ngẫu nhiên lấy mẫu, không có hiệu ứng nhân quả nào.

### 6.3. Trực quan hóa

Histogram của 200 giá trị chênh lệch, với:
- Đường thẳng đứng tại `0` (không chênh lệch - độc lập hoàn hảo).
- Đường đứt nét tại giá trị chênh lệch của LẦN CHẠY CHÍNH (quần thể lớn) cho thấy trực quan rằng kết quả "chính" hoàn toàn nằm gọn trong biên độ dao động bình thường của phân phối null, không phải một điểm bất thường (outlier).

Đo thực tế: qua 200 lần chạy, chênh lệch dao động quanh **+0.12 điểm %** (độ lệch chuẩn ~1.6) - một kết quả đơn lẻ dương hay âm vài điểm % là hoàn toàn bình thường trong biên độ nhiễu này.

## 7. Đối chứng: khi THẬT SỰ có hiệu ứng (`true_effect ≠ 0`)

Để kiểm tra tính đúng đắn của toàn bộ pipeline (sanity check), có thể đặt
`true_effect = 0.3` (cộng thêm 30 điểm % vào xác suất gặp họa CỦA RIÊNG người trúng số). Khi đó:

- `P(B|A)` sẽ cao hơn hẳn `P(B|not A) + 0.1` (kiểm chứng trong
  `test_true_effect_creates_detectable_difference`).
- `p_value < 0.01` - Chi-square test THÀNH CÔNG phát hiện được liên hệ. `co_y_nghia_thong_ke = True`.

Điều này xác nhận: pipeline có đủ **độ nhạy (statistical power)** để phát hiện một hiệu ứng thật khi nó tồn tại - kết quả "không có ý nghĩa thống kê" ở kịch bản mặc định không phải vì công cụ kiểm định yếu, mà vì đúng là không có hiệu ứng nào để phát hiện.

## 8. Độ phức tạp

| Bước | Độ phức tạp |
|-|-|
| Sinh quần thể (n người) | O(n) - vector hóa hoàn toàn |
| Xây bảng 2×2 | O(n) - 4 phép AND/NOT vector hóa |
| Chi-square test | O(1) - bảng 2×2 cố định |
| Repeated trials (k lần, m người/lần) | O(k × m) |

Đo thực tế: quần thể chính 2 triệu người + 200 lần chạy null (200.000 người/lần) - toàn bộ pipeline chạy trong **~0.6 giây** trên CPU 8 luồng ~2.5GHz.

## 9. Vì sao không dùng dataset "Lottery Curse" thật ?

Bí kíp gốc có đề cập mượn dataset công khai về tỷ lệ phá sản/ly hôn/tai nạn của người trúng Powerball (Mỹ/Âu). Quyết định thiết kế ở đây là KHÔNG dùng số liệu thật chưa kiểm chứng được nguồn gốc, vì 2 lý do:

1. **Rủi ro lan truyền sai lệch:** dùng một con số "nghe có vẻ khoa học" mà không kiểm chứng được nguồn là nguy hiểm hơn không có số liệu gì.
2. **Không cần thiết cho mục đích giáo dục:** câu hỏi cốt lõi ("A, B độc lập có tạo ra chênh lệch quan sát được không") trả lời được đầy đủ bằng mô phỏng có kiểm soát - vì ta BIẾT chính xác cơ chế sinh dữ liệu (từ đó biết chắc H0 đúng), nên có thể kiểm chứng công cụ thống kê (Chi-square, CI, phân phối null) hoạt động đúng như lý thuyết dự đoán.

## 10. Tham chiếu

- Pearson, K. (1900). *On the criterion that a given system of deviations... is such that it can be reasonably supposed to have arisen from random sampling* - nguồn gốc phép kiểm định Chi-square.
- Haldane, J. B. S. (1956) / Anscombe, F. J. (1956) - hiệu chỉnh liên tục cho odds ratio khi có ô count = 0.
- Taleb, N. N. (2007). *The Black Swan* - khái niệm "narrative fallacy" (ngụy biện tường thuật), nền tảng triết lý cho phần 6 của tài liệu này.
- Kahneman, D., & Tversky, A. (1974). *Judgment under Uncertainty: Heuristics and Biases* - thiên kiến sẵn có (availability heuristic), giải thích vì sao con người dễ nhớ và tin vào các câu chuyện "trúng số rồi gặp họa" hơn là nền tảng thống kê đầy đủ.
