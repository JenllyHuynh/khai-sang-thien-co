# Tài Liệu Kỹ Thuật - Trận Pháp 1: Vạn Kiếp Quy Tông (Monte Carlo Simulator)

## 1. Bài toán cần giải

Xổ số 6/55: chọn 6 số phân biệt trong khoảng 1..55. Tổng số tổ hợp có thể:

```
C(55, 6) = 55! / (6! × 49!) = 28.989.675
```

Bí kíp gốc đặt câu hỏi: **"Chọn số theo ngày sinh (số đẹp) có làm tăng xác suất trúng không?"** Câu trả lời toán học là **không** . 
Vì kết quả quay số là phân bố đều (uniform) trên toàn bộ 28.989.675 tổ hợp, bất kể người chơi chọn số gì. Xác suất trúng của MỘT vé bất kỳ luôn là:

```
P(trúng) = 1 / C(55,6) ~ 3.449 × 10^-8
```

Câu hỏi thật sự đáng mô phỏng không phải "có trúng không" (vì việc chờ rúng thật trong vài triệu vé mô phỏng là bất khả thi - cần trung bình 29 triệu vé mới có 1 vé trúng), mà là: **"Nếu một tổ hợp trúng, có bao
nhiêu người khác cũng chọn đúng tổ hợp đó (phải chia giải) ?"** 

Đây là bài toán "collision" (đụng độ) kiểu Birthday Paradox, và nó HOÀN TOÀN đo được với vài triệu vé - đây chính là nội dung mô phỏng của trận pháp.

## 2. Kiến trúc pipeline

```
lottery_config.py      ->  hằng số luật chơi + trọng số thiên vị
ticket_generator.py    ->  sinh vé (vector hóa 100%)
encode.py              ->  mã hóa vé -> id int64 duy nhất
simulate.py            ->  điều phối sinh vé theo lô (chunk)
stats.py               ->  đếm trùng lặp, thống kê cụm
visualize.py            ->  vẽ biểu đồ
run.py / run_repeated.py -> CLI (1 lần / lặp nhiều lần)
aggregate.py            ->  tổng hợp nhiều lần chạy độc lập
```

Dữ liệu chảy qua pipeline như sau:

``` text
weights (55,) ----+
                   |
                   v
rng ----------> generate_tickets() --> tickets (n, 6) uint8 --> encode_tickets() --> ids (n,) int64
                                                                                          |
                                                                                          v
                                                                                   np.unique(ids) --> cluster_stats()
```

## 3. Sinh vé có trọng số - Efraimidis–Spirakis weighted sampling

### 3.1. Vì sao không dùng vòng lặp qua từng vé ?

Cách "ngây thơ" để sinh 1 vé có trọng số (ưu tiên số <= 31) là dùng `numpy.random.Generator.choice(..., p=weights, replace=False)` trong một vòng lặp Python - nhưng với hàng triệu vé, vòng lặp Python (chậm ~10^7 lần/giây tốt nhất) sẽ mất hàng chục giây đến hàng phút, và không tận dụng được SIMD/vector hóa của numpy.

### 3.2. Thuật toán

Với mỗi số `i` trong 1..55 có trọng số `w_i`, sinh:

```
U_i ~ Uniform(0, 1)          (độc lập cho mỗi số)
key_i = U_i ^ (1 / w_i)
```

**PICK = 6 số có `key` lớn nhất** chính là mẫu có trọng số, không lặp lại, theo đúng phân phối xác suất tỷ lệ với `w_i` (chứng minh: Efraimidis & Spirakis, "Weighted random sampling with a reservoir", 2006). Trực giác: số có trọng số cao -> số mũ `1/w_i` nhỏ -> `U^(1/w_i)` có xu hướng gần 1 hơn -> dễ lọt vào top-6.

### 3.3. Cài đặt vector hóa (`ticket_generator.py`)

```python
u = rng.random((n, N_NUMBERS), dtype=np.float32)         # (n, 55)
u = np.clip(u, 1e-12, 1.0)                               # tránh log(0)
keys = u ** (1.0 / weights)                              # broadcast (n,55) ** (55,)
top_idx = np.argpartition(-keys, PICK - 1, axis=1)[:, :PICK]
numbers = (top_idx + 1).astype(np.uint8)
```

- **Một phép toán ma trận duy nhất** sinh ra `n` vé cùng lúc - không có for-loop Python nào chạy qua từng vé.
- `np.argpartition` (độ phức tạp trung bình O(55) mỗi hàng, tổng O(n × 55)) được dùng thay vì `np.argsort` (O(n × 55 log 55)) vì ta chỉ cần TOP-6, không cần thứ tự đầy đủ của cả 55 số - nhanh hơn đáng kể.
- `dtype=np.float32` giảm một nửa bộ nhớ so với `float64` mặc định, và đủ độ chính xác cho mục đích lấy mẫu (không cần độ chính xác tuyệt đối của phép tính khoa học).

### 3.4. Hai bộ trọng số

| Phe          | Công thức trọng số                                                            | Ý nghĩa                        |
| ------------ | ----------------------------------------------------------------------------- | ------------------------------ |
| **Vô Vi**    | `w_i = 1` với mọi i                                                           | Phân phối đều tuyệt đối        |
| **Tình Cảm** | `w_i = 3` nếu `i <= 31`, `w_i = 6` nếu `i ∈ {1,9,11,20,36}`, còn lại `w_i = 1` | Thiên vị số ngày sinh + số hên |

(`BiasConfig` trong `lottery_config.py` cho phép chỉnh các trọng số này.)

## 4. Mã hóa vé thành số nguyên duy nhất (`encode.py`)

### 4.1. Vấn đề

Sau khi có hàng triệu vé (mỗi vé là 6 số), cần đếm xem có bao nhiêu cặp vé TRÙNG NHAU (cùng 6 số, không quan tâm thứ tự). Cách ngây thơ - so sánh từng cặp vé - là O(n^2), với n = 2 triệu sẽ là 4×10^12 phép so sánh: không khả thi.

### 4.2. Giải pháp: đóng gói thành 1 số nguyên (positional encoding)

Với 6 số đã sort tăng dần `(a₁ < a₂ < ... < a₆)`, mỗi số nằm trong `[1, 55]`. Chọn cơ số `BASE = 56` (lớn hơn giá trị lớn nhất có thể), mã hóa như một số ở hệ cơ số 56:

```
id = a1 × 56^5 + a2 × 56^4 + a3 × 56^3 + a4 × 56^2 + a5 × 56^1 + a6 × 56^0
```

Giá trị lớn nhất có thể: `55 × (56^6 - 1)/(56 - 1) ~ 3.07 × 10^10` - nằm gọn trong `int64` (giới hạn ~9.2 × 10^18), không có nguy cơ tràn số.

Cài đặt (vector hóa, Horner's method):
```python
sorted_tickets = np.sort(tickets, axis=1).astype(np.int64)
ids = np.zeros(n, dtype=np.int64)
for col in range(6):                    # chỉ lặp 6 lần (hằng số!), KHÔNG lặp qua n vé
    ids = ids * BASE + sorted_tickets[:, col]
```

Vòng lặp ở đây chạy đúng **6 lần** (số cột), không phụ thuộc vào `n` - mỗi lần là một phép toán vector hóa trên toàn bộ `n` vé cùng lúc. Độ phức tạp tổng: O(n), với hằng số rất nhỏ (6 phép nhân + cộng element-wise).

### 4.3. Đếm trùng lặp: O(n log n)

Với mảng `ids` (n,) int64, `np.unique(ids, return_counts=True)` (dùng sort nội bộ, O(n log n)) trả về danh sách id duy nhất kèm số lần xuất hiện - đây chính là "cụm" (cluster) các vé trùng nhau.

## 5. Chia lô (chunking) - kiểm soát RAM

### 5.1. Vấn đề bộ nhớ

Ma trận trung gian `u` trong bước sinh vé có kích thước `(n, 55)` `float32` = `n × 220` byte. Với `n = 10.000.000`, đó là **2.2 GB** chỉ cho MỘT ma trận tạm - có thể làm cạn RAM trên máy 8GB khi cộng thêm hệ điều hành + các mảng khác.

### 5.2. Giải pháp

`simulate.py` sinh vé theo từng lô (`chunk_size`, mặc định 200.000), với mỗi lô:
1. Sinh ma trận `u` kích thước `(chunk_size, 55)` - chỉ ~44MB.
2. Mã hóa ngay thành `ids` (int64, rất nhẹ - 8 byte/vé).
3. **Giải phóng `u` và `tickets`** (Python GC thu hồi khi hết tham chiếu) trước khi sang lô tiếp theo.
4. Nối các mảng `ids` của từng lô bằng `np.concatenate` ở cuối.

Nhờ vậy, **RAM đỉnh không phụ thuộc vào tổng số vé muốn sinh**, chỉ phụ thuộc vào `chunk_size` - đo thực tế: RAM đỉnh **~350MB** dù sinh 4 triệu vé (2 phe × 2 triệu), và tăng tuyến tính về THỜI GIAN chứ không phải RAM khi tăng số vé.

---

## 6. Thống kê cụm trùng lặp (`stats.py`)

Với mảng `unique` (các id duy nhất) và `counts` (số lần mỗi id xuất hiện), các chỉ số được tính:

| Chỉ số | Công thức |
|---|---|
| `unique_ratio` | `n_unique / n_players` |
| `pct_players_sharing_ticket` | `100 × Σ(counts[counts>1]) / n_players` |
| `max_cluster_size` | `max(counts)` |
| `mean_cluster_size_if_shared` | `mean(counts[counts>1])` |

`pct_players_sharing_ticket` chính là chỉ số cốt lõi: **% người chơi mà vé của họ trùng với ít nhất 1 người khác**. Đo thực tế: Phe Vô Vi ~6.7%, Phe Tình Cảm ~27.4% (gấp ~4-6 lần), dù xác suất trúng lý thuyết của cả hai phe là như nhau.

---

## 7. Chạy nhiều lần độc lập (`run_repeated.py` + `aggregate.py`)

### 7.1. Động lực

Một lần chạy đơn lẻ (dù n lớn) vẫn chỉ là MỘT mẫu từ một seed. Để có bằng chứng thống kê vững chắc rằng hiện tượng "Tình Cảm trùng vé nhiều hơn" không phải ngẫu nhiên do may rủi của 1 seed, cần lặp lại với nhiều seed độc lập và xem phân phối kết quả.

### 7.2. Thiết kế: streaming append, không tích lũy file

Mỗi lần chạy (`_run_one_trial`) chỉ trả về **các con số tóm tắt** (8 giá trị scalar), được ghi thành **1 dòng CSV** ngay lập tức (`writer.writerow` + `f.flush()`), rồi giải phóng toàn bộ mảng `ids` trước khi sang lần
chạy tiếp theo. Điều này giữ cho:

- **Bộ nhớ**: không tích lũy dữ liệu thô qua các lần chạy - mỗi lần chạy độc lập về bộ nhớ với lần trước.
- **An toàn khi ngắt giữa chừng**: `f.flush()` sau mỗi dòng đảm bảo dữ liệu đã ghi không mất khi tắt máy/Ctrl+C. Cơ chế resume (`_existing_run_count`) đếm số dòng đã có trong CSV để biết chạy tiếp từ `run_index` nào, với `seed = seed_start + run_index` - đảm bảo không bao giờ chạy trùng seed.

### 7.3. Tổng hợp có trọng số (`aggregate.py`)

Vì các phiên chạy khác nhau có thể dùng `n_players_per_group` khác nhau (người dùng đổi cấu hình giữa các lần chạy), việc tổng hợp dùng **trung bình có trọng số**:

```python
weighted_mean = Σ(pct_sharing_i × n_players_i) / Σ(n_players_i)
```

thay vì trung bình đơn giản (sẽ đánh giá sai nếu các lần chạy có cỡ mẫu khác nhau nhiều).

## 8. Độ phức tạp tổng thể

| Bước                        | Độ phức tạp thời gian | Độ phức tạp bộ nhớ                         |
| --------------------------- | --------------------- | ------------------------------------------ |
| Sinh vé (1 lô)              | O(chunk_size × 55)    | O(chunk_size × 55)                         |
| Mã hóa vé                   | O(n)                  | O(n)                                       |
| Đếm trùng lặp (`np.unique`) | O(n log n)            | O(n)                                       |
| **Tổng (1 lần chạy n vé)**  | **O(n log n)**        | **O(chunk_size × 55)** - không phụ thuộc n |

Đo thực tế trên CPU 8 luồng ~2.5GHz: **2 triệu vé/phe trong ~5.6 giây**, RAM đỉnh ~350MB.

## 9. Tham chiếu

- Efraimidis, P. S., & Spirakis, P. G. (2006). *Weighted random sampling with a reservoir*. Information Processing Letters, 97(5), 181-185.
- Bài toán Birthday Paradox (đụng độ ngẫu nhiên) - nền tảng toán học cho phần "trùng vé" của trận pháp.
