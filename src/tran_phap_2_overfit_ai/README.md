# Trận Pháp 2: Lão Tặc AI - Kẻ Học Vẹt (Overfitted Prophet)

> _"AI tưởng mình thông thiên, nhưng trước Random chỉ là một thằng học vẹt."_

## Ý tưởng

Huấn luyện 2 "đạo sĩ" AI trên cùng một bộ lịch sử quay số giả lập, rồi đối chiếu ai thành thật, ai ảo tưởng:

- **Lão Tặc AI (Học Vẹt)** - cây quyết định không giới hạn độ sâu, CỐ TÌNH được cho xem `draw_index` (số thứ tự lần quay) và `number` làm đặc trưng. Vì cặp này là **định danh duy nhất** của mỗi dòng dữ liệu (không phải tín hiệu dự đoán thật), nó có thể "học thuộc lòng" toàn bộ đáp án lịch sử.
- **Đạo Sĩ Khiêm Tốn (Baseline)** - cây quyết định bị giới hạn độ sâu, KHÔNG được xem định danh, chỉ có tần suất/khoảng cách xuất hiện gần đây.

Cả hai được đánh giá trên 2 tập: dữ liệu **đã học** (train) và dữ liệu **tương lai chưa từng thấy** (test, chia theo mốc thời gian - không xáo trộn ngẫu nhiên, giữ đúng tinh thần "học quá khứ, thi tương lai").

## Chân lý hé lộ (chạy thật, 1560 lần quay ~10 năm)

| | Trên dữ liệu ĐÃ HỌC | Trên TƯƠNG LAI chưa thấy |
|---|---|---|
|  Lão Tặc AI | 6.000/6 số trúng, tự tin 100% | **0.636/6 số trúng** (còn tệ hơn đoán mò!), **vẫn tự tin 100%** |
|  Đạo Sĩ Khiêm Tốn | 0.702/6 số trúng, tự tin 11.4% | 0.523/6 số trúng, tự tin 10.9% |

Đoán mò lý thuyết (random 6/55): **0.655/6 số trúng**.

**Điểm chí mạng:** Lão Tặc AI không chỉ dự đoán sai trên tương lai. Nó **vẫn tự tin 100%** trong lúc sai! 
Vì các lá (leaf) của cây được huấn luyện với `min_samples_leaf=1` thường chỉ chứa 1 mẫu duy nhất -> xác suất đầu ra gần như luôn là 0 hoặc 1 tuyệt đối, bất kể lá đó đúng hay sai khi gặp dữ liệu mới. Đây chính xác là hình ảnh "ảo tưởng sụp đổ trước sự vô thường của Random" mà bí kíp gốc muốn khắc họa - và biểu đồ có hẳn 1 "ngôi sao vàng" (sự thật trần trụi) đặt chồng lên cột độ tự tin để lộ rõ khoảng cách này.

## Chạy thử

```bash
pip install -r requirements.txt

python -m src.tran_phap_2_overfit_ai.run
```

Tinh chỉnh:
```bash
# Lịch sử ngắn hơn/dài hơn (đơn vị: số lần quay)
python -m src.tran_phap_2_overfit_ai.run --n-draws 3000

# Cửa sổ tính tần suất gần đây (mặc định 52 ~ 1 năm nếu quay 1 lần/tuần)
python -m src.tran_phap_2_overfit_ai.run --window 30

# Tỷ lệ dữ liệu cuối dùng làm "tương lai" kiểm tra
python -m src.tran_phap_2_overfit_ai.run --test-frac 0.2
```

Kết quả lưu tại `outputs/tran_phap_2/`:
- `hoc_vet_vs_thuc_te.png` - biểu đồ so sánh 2 model, 2 tập dữ liệu
- `tong_ket.json` - số liệu đầy đủ (bao gồm `ao_tuong_gap` - khoảng cách giữa độ tự tin AI tự nhận và sự thật)

Với quy mô mặc định (1560 lần quay), toàn bộ pipeline chạy trong khoảng **3 giây** trên máy CPU 8 luồng ~2.5GHz / 8GB RAM - không cần lo hiệu năng.

## Vì sao không dùng dữ liệu quay số thật ?

Như "Kiếp Nạn 2" trong bí kíp gốc đã lường trước, không có dữ liệu công khai đáng tin cậy về lịch sử quay số Việt Nam. Nhưng thật ra điều đó KHÔNG quan trọng với trận pháp này: vì bản chất một lần quay số thật CŨNG LÀ random thuần túy (nếu máy quay số vận hành đúng), lịch sử giả lập bằng `numpy` độc lập-đồng nhất (IID) mô phỏng đúng bản chất thống kê của dữ liệu thật - không hề "kém chân thực" hơn, chỉ là không có màu mè lịch sử thật. Đây chính là cái hay của trận pháp: chứng minh AI ảo tưởng ngay cả khi dữ liệu HOÀN TOÀN không có quy luật ẩn nào - không cần biện minh rằng "vì dữ liệu giả nên AI mới thua".

## Kỹ thuật đứng sau

1. **Tái sử dụng bộ sinh số của Trận Pháp 1** (`generate_uniform`) để tạo lịch sử quay số - đảm bảo nhất quán logic "random thuần túy" xuyên suốt cả bí kíp.
2. **Xây đặc trưng vector hóa 100%** bằng kỹ thuật prefix-sum (tính tần suất trong cửa sổ trượt) và cumulative-max (tính khoảng cách tới lần xuất hiện gần nhất) - không có Python for-loop chạy qua từng cặp (lần quay, số), dù bảng đặc trưng có thể lên tới hàng chục nghìn dòng.
3. **Chia tập train/test theo THỜI GIAN**, không xáo trộn ngẫu nhiên - đúng tinh thần "học quá khứ, kiểm tra trên tương lai chưa từng thấy", tránh rò rỉ thông tin tương lai vào lúc tính đặc trưng (mọi đặc trưng tại thời điểm t chỉ dùng dữ liệu từ trước t).
4. **Cơ chế overfit có chủ đích**: `DecisionTreeClassifier(max_depth=None, min_samples_leaf=1)` + đặc trưng `draw_index`/`number` (định danh duy nhất mỗi dòng) -> cây có đủ "tự do" để tạo 1 lá riêng cho mỗi điểm dữ liệu train, ghi nhớ tuyệt đối mà không học được quy luật tổng quát nào (vì không có quy luật nào để học).
