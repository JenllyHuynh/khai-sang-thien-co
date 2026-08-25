# Tài Liệu Kỹ Thuật - Trận Pháp 2: Lão Tặc AI, Kẻ Học Vẹt (Overfitted Prophet)

## 1. Bài toán cần giải

Chứng minh bằng thực nghiệm rằng một mô hình Machine Learning có thể đạt độ chính xác gần tuyệt đối trên dữ liệu lịch sử (train), NHƯNG vô dụng hoàn toàn khi dự đoán tương lai - nếu bản thân dữ liệu không có quy luật thật để học (như xổ số, vốn là random thuần túy). Đây là minh họa kinh điển của **overfitting**, đặc biệt là dạng nguy hiểm nhất: overfitting thông qua **rò rỉ định danh** (identity leakage).

## 2. Kiến trúc pipeline

```
data_generator.py   ->  sinh lịch sử quay số giả lập (tái dùng Trận Pháp 1)
features.py         ->  xây bảng đặc trưng dạng "long format", vector hóa 100%
overfit_model.py     ->  2 model đối chứng: Lão Tặc AI (overfit) vs Đạo Sĩ Khiêm Tốn (baseline)
simulate.py           ->  chia train/test theo thời gian, huấn luyện, đánh giá
stats.py              ->  tổng hợp: số trúng trung bình, độ tự tin, "khoảng cách ảo tưởng"
visualize.py          ->  biểu đồ 2 panel (số trúng + ảo tưởng tự tin)
run.py                ->  CLI
```

---

## 3. Sinh dữ liệu lịch sử (`data_generator.py`)

Tái sử dụng trực tiếp `generate_uniform()` từ Trận Pháp 1 (Phe Vô Vi). Vì một lần quay số THẬT về bản chất thống kê cũng chính là random thuần túy, không khác gì cách Phe Vô Vi chọn số. Điều này đảm bảo:

1. **Nhất quán logic** xuyên suốt cả bí kíp (không định nghĩa lại RNG).
2. **Không có "quy luật ẩn" nào được cài cắm** vào dữ liệu lịch sử - đây là điều kiện tiên quyết để bài học overfitting có ý nghĩa (nếu dữ liệu CÓ quy luật thật, một mô hình học được nó không phải overfitting).

```python
draws = generate_uniform(n_draws, rng)   # (n_draws, 6) - chưa sort
return np.sort(draws, axis=1)             # sort mỗi hàng, theo thời gian
```

## 4. Xây đặc trưng (`features.py`) - vector hóa hoàn toàn

### 4.1. Cấu trúc dữ liệu "long format"

Với `n_draws` lần quay và 55 số, ta cần một bảng mà mỗi dòng là 1 cặp `(lần quay t, số n)`, với nhãn `label = 1` nếu số `n` xuất hiện ở lần quay `t`. Kích thước: `(n_draws - window) × 55` dòng.

### 4.2. Ba đặc trưng thống kê (không rò rỉ tương lai)

| Đặc trưng | Công thức | Ý nghĩa |
|---|---|---|
| `freq_last_window` | `Σ presence[t-window : t, n] / window` | Tần suất xuất hiện trong `window` lần quay gần nhất |
| `overall_freq` | `Σ presence[0 : t, n] / t` | Tần suất lũy kế từ đầu lịch sử |
| `gap_since_last` | `min(t - last_seen_before_t, window) / window` | Khoảng cách (chuẩn hóa) tới lần xuất hiện gần nhất |

**Nguyên tắc chống rò rỉ (data leakage):** mọi đặc trưng tại thời điểm `t` CHỈ dùng dữ liệu từ `[0, t)` - không bao giờ dùng `presence[t]` (kết quả của chính lần quay đang dự đoán) để tính đặc trưng cho lần quay đó.

### 4.3. Vector hóa bằng prefix-sum và cumulative-max

**`freq_last_window` và `overall_freq`** - dùng kỹ thuật **prefix-sum**:

```python
cumsum = vstack([zeros(1,55), presence.cumsum(axis=0)])   # cumsum[t] = Σ presence[0:t]
freq_last_window[t] = (cumsum[t] - cumsum[t-window]) / window   # O(1) mỗi t, thay vì O(window)
overall_freq[t] = cumsum[t] / t
```

Đây là kỹ thuật "sliding window sum" cổ điển: thay vì tính lại tổng của `window` phần tử mỗi lần (O(window) mỗi bước, O(n×window) tổng), ta chỉ cần **một phép trừ 2 giá trị prefix-sum** (O(1) mỗi bước, O(n) tổng).

**`gap_since_last`** - dùng kỹ thuật **cumulative-max**:

```python
occurred_idx = where(presence == 1, arange(n_draws)[:,None], -1)   # (n_draws, 55)
cummax = maximum.accumulate(occurred_idx, axis=0)                    # cummax[t] = lần gần nhất <=t đã xuất hiện
last_seen_before = vstack([full((1,55),-1), cummax[:-1]])            # dịch xuống 1 hàng (chỉ tính t' < t)
```

`np.maximum.accumulate` tính "giá trị lớn nhất tính đến vị trí hiện tại" theo trục cho trước trong O(n) - đây chính xác là "lần xuất hiện gần nhất" nếu ta mã hóa "không xuất hiện" là `-1` (luôn nhỏ hơn mọi index hợp lệ >= 0).

**Kết quả:** toàn bộ `n_draws × 55` đặc trưng được tính trong O(n_draws) với hằng số rất nhỏ - KHÔNG có Python for-loop nào chạy qua từng cặp (lần quay, số), dù bảng cuối có thể lên tới hàng chục nghìn dòng.

## 5. Cơ chế overfitting có chủ đích (`overfit_model.py`)

### 5.1. Hai bộ đặc trưng

```python
FEATURES_OVERFIT  = ["draw_index", "number", "freq_last_window", "overall_freq", "gap_since_last"]
FEATURES_BASELINE = [                          "freq_last_window", "overall_freq", "gap_since_last"]
```

Điểm khác biệt duy nhất: **Lão Tặc AI được cho xem `draw_index` và `number`**.

### 5.2. Vì sao đây là "cho xem đáp án" ?

Cặp `(draw_index, number)` là **định danh duy nhất** của mỗi dòng dữ liệu - không phải tín hiệu dự đoán thật. Với một `DecisionTreeClassifier` không giới hạn độ sâu (`max_depth=None, min_samples_leaf=1`), cây có thể tiếp tục chia nhỏ (split) trên 2 trục tọa độ liên tục này cho đến khi **mỗi điểm dữ liệu train nằm trong một lá (leaf) riêng của chính nó** - về mặt hình học, đây tương đương một cấu trúc kiểu k-d tree: với đủ độ sâu, luôn tồn tại một chuỗi ngưỡng (threshold) trên `draw_index` và `number` cô lập được BẤT KỲ điểm nào khỏi phần còn lại, vì mỗi cặp `(draw_index, number)` là duy nhất trong toàn bộ tập dữ liệu.

**Kết quả:** cây "học thuộc lòng" đáp án của từng dòng lịch sử - không học được quy luật tổng quát nào (vì làm gì có quy luật thật trong random thuần túy), chỉ đơn thuần **ghi nhớ**.

### 5.3. Đạo Sĩ Khiêm Tốn - baseline kiểm soát

```python
DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
```

Không có `draw_index`/`number`, cây bị giới hạn độ sâu 3 và mỗi lá phải có >=50 mẫu - không đủ "tự do" để ghi nhớ, chỉ có thể học các xu hướng tần suất/khoảng cách rất thô (mà thực chất cũng không mang tín hiệu dự đoán thật, vì dữ liệu là random). Baseline này đóng vai trò đối chứng: chứng minh rằng KHÔNG PHẢI cứ dùng Decision Tree là overfit - chính việc cho xem định danh + độ sâu không giới hạn mới là nguyên nhân.

### 5.4. Đánh giá: "Top-6 prediction"

Với mỗi lần quay, lấy xác suất `P(label=1)` mà mô hình gán cho cả 55 số, chọn **6 số có xác suất cao nhất** làm "dự đoán của AI", so khớp với 6 số thật:

```python
proba = model.predict_proba(X)[:, index_of_class_1]
top6 = nlargest(6, proba)
so_khop = count(top6 ∩ actual_numbers)   # 0 đến 6
```

Đường cơ sở lý thuyết (đoán ngẫu nhiên 6 số trong 55, so với 6 số thật):

```
E[số trúng] = PICK × PICK / N_NUMBERS = 6 × 6 / 55 ~ 0.6545
```

## 6. Chia train/test theo thời gian (`simulate.py`)

```python
split_at = unique_draw_idx[int(n_usable × (1 - test_frac))]
df_train = df[df.draw_index <  split_at]
df_test  = df[df.draw_index >= split_at]
```

**Không xáo trộn ngẫu nhiên (no shuffle)** - đúng tinh thần "học quá khứ, kiểm tra trên tương lai chưa từng thấy". Đây cũng là lý do `draw_index` của tập test luôn LỚN HƠN mọi giá trị `draw_index` mà mô hình từng thấy lúc train - khiến việc "ghi nhớ" của Lão Tặc AI hoàn toàn vô nghĩa trên tập test (nó phải ngoại suy các ngưỡng threshold ra ngoài phạm vi đã học, cho kết quả về cơ bản là tùy tiện).

## 7. "Khoảng cách ảo tưởng" (ao_tuong_gap) - phát hiện thú vị nhất

### 7.1. Hiện tượng quan sát được

Khi chạy thật, Lão Tặc AI **vẫn giữ độ tự tin ~100%** trên tập test, dù số trúng trung bình rơi xuống dưới cả mức đoán mò. Đây KHÔNG phải lỗi - là hệ quả tất yếu của `min_samples_leaf=1`: mỗi lá của cây gần như luôn chỉ chứa **1 mẫu train duy nhất**, nên xác suất đầu ra của lá đó là tuyệt đối `0.0` hoặc `1.0` (không có sự "pha trộn" của nhiều mẫu để tạo ra một xác suất trung gian). Khi một điểm dữ liệu MỚI (test) rơi vào bất kỳ lá nào (dù không liên quan gì đến logic đã học), nó vẫn nhận được xác suất tuyệt đối đó - mô hình "tự tin" dù đang đoán bừa.

### 7.2. Công thức

```python
real_accuracy = avg_matches / PICK                 # tỷ lệ THỰC SỰ đúng
ao_tuong_gap  = avg_confidence - real_accuracy       # độ tự tin AI tự nhận − sự thật
```

Đo thực tế (1560 lần quay, ~10 năm): Lão Tặc AI có `ao_tuong_gap` trên tập test là **+89.4 điểm %** (tự tin 100%, thực tế chỉ đúng ~10.6%), trong khi Đạo Sĩ Khiêm Tốn chỉ **+2.2 điểm %** (gần trung thực).

### 7.3. Trực quan hóa: ngôi sao "sự thật trần trụi"

Biểu đồ cột thể hiện độ tự tin (%), với một **marker hình sao** đặt chồng lên ở độ cao = `real_accuracy × 100` - khoảng cách trực quan giữa đỉnh cột (ảo tưởng) và vị trí ngôi sao (sự thật) chính là hình ảnh cô đọng nhất của bài học overfitting trong toàn bộ trận pháp.

## 8. Độ phức tạp

| Bước | Độ phức tạp |
|---|---|
| Sinh lịch sử | O(n_draws) (vector hóa, tái dùng Trận Pháp 1) |
| Xây đặc trưng | O(n_draws) (prefix-sum + cumulative-max, không loop lồng) |
| Huấn luyện Decision Tree | O(n_samples × n_features × log(n_samples)) trung bình |
| Đánh giá top-6 | O(n_draws × 55 log 55) (groupby + nlargest) |

Đo thực tế: **1560 lần quay lịch sử (~10 năm), toàn bộ pipeline (sinh dữ liệu -> xây đặc trưng -> huấn luyện 2 model -> đánh giá 4 tổ hợp) chạy trong ~3.2 giây** trên CPU 8 luồng ~2.5GHz.

## 9. Tại sao không cần dữ liệu quay số thật ?

Vì bản chất một lần quay số thật (nếu máy quay vận hành đúng) CŨNG LÀ random thuần túy - dữ liệu giả lập bằng `numpy` độc lập-đồng nhất (IID) mô phỏng đúng bản chất thống kê của dữ liệu thật. Cái hay của trận pháp nằm ở việc chứng minh AI ảo tưởng ngay cả khi dữ liệu HOÀN TOÀN không có quy luật ẩn nào - không cần biện minh "vì dữ liệu giả nên AI mới thua".

## 10. Tham chiếu

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* - Chương 7: Model Assessment and Selection (bias-variance tradeoff, overfitting).
- Khái niệm "data leakage" trong Machine Learning - đặc biệt dạng rò rỉ qua định danh/chỉ số thời gian, một trong những lỗi phổ biến và khó phát hiện nhất trong thực hành ML.
