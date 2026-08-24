# Trận Pháp 3: Phá Giải Hỗn Mang (Chaos vs. PRNG)

> _"Hỗn mang có quy luật, nhưng Xổ Số là bậc thầy của vô thường."_

## Ý tưởng

Trận pháp này tách bạch 2 khái niệm hay bị nhầm lẫn: **"trông hỗn loạn"** và **"thực sự khó đoán"** - không phải cứ trông rối rắm là an toàn để làm bộ sinh số cho xổ số.

### Phần 1: Hỗn Mang Có Quy Luật (Lorenz Attractor)

**Hệ phương trình Lorenz** (khí tượng học, 1963) là "hỗn loạn tất định": công thức hoàn toàn không có gì ngẫu nhiên, nhưng quỹ đạo trông như random, không lặp lại, và cực kỳ nhạy với điều kiện ban đầu. Sai khác`1e-8` ở điểm khởi tạo, sau vài chục đơn vị thời gian sẽ cho quỹ đạo hoàn toàn khác - "hiệu ứng cánh bướm" kinh điển, mô phỏng và đo tốc độ phân kỳ (số mũ Lyapunov) bằng chính code trong repo này.

**Điểm mấu chốt:** dù nhìn hỗn loạn, hệ Lorenz vẫn là một **hàm số tất định** - nếu biết chính xác điều kiện ban đầu, quỹ đạo tương lai hoàn toàn xác định (không hề "ngẫu nhiên thật"). Đây chính là ẩn dụ cho việc một máy quay số vật lý (bi lăn, luồng khí...) về bản chất vẫn là hệ tất định - thứ khiến nó trở nên khó đoán trong thực tế là hỗn loạn + giới hạn đo lường, không phải "ngẫu nhiên thuần khiết" theo nghĩa triết học.

### Phần 2: Bắt Mạch Yêu Quái (LCG yếu vs CSPRNG mạnh)

So sánh trực diện 2 bộ sinh số:
- **LCG yếu (kiểu RANDU, IBM 1968)** - PRNG khét tiếng tệ nhất lịch sử.
- **CSPRNG mạnh (PCG64, mặc định của numpy)** - chính là bộ sinh số đứng sau toàn bộ Trận Pháp 1 & 2 của bí kíp này.

Hai màn trình diễn:
1. **Scatter 3D của bộ ba số liên tiếp**, nhìn đúng góc toán học (tính từ pháp tuyến của họ mặt phẳng Marsaglia): LCG yếu lộ rõ **15 dải "siêu phẳng" song song**, còn CSPRNG mạnh là một khối mây đặc không cấu trúc.
2. **"Bắt mạch" (state reconstruction)**: chỉ với **1 số quan sát**, ta dựng lại được toàn bộ trạng thái nội bộ của LCG yếu và dự đoán tuyệt đối chính xác mọi số tiếp theo (sai số = 0.00). Thử đúng kỹ thuật đó lên CSPRNG mạnh - thất bại hoàn toàn (sai số ~0.5-1.0, như đoán mò).

## Chân lý hé lộ (chạy thật)

```
    Hiệu ứng cánh bướm: sai số khởi tạo 1e-8 nhân đôi mỗi ~1.17 đơn vị thời gian
   (số mũ Lyapunov ước lượng ~ 0.59; giá trị lý thuyết tham khảo ~ 0.9056)

    Bắt mạch LCG yếu (RANDU): chỉ 1 số quan sát, dự đoán 4999 số tiếp theo
   -> sai số tối đa: 0.00e+00 (CRACK THÀNH CÔNG)
    Thử ĐÚNG kỹ thuật đó lên CSPRNG mạnh (PCG64):
   -> sai số tối đa: 9.89e-01 (THẤT BẠI HOÀN TOÀN)
```

**Kết luận của bí kíp:** muốn đoán trúng xổ số dùng CSPRNG thật, phải biết trạng thái nội bộ/seed của nó - chuyện đó là **"tà tu"** (tấn công mật mã học ở trình độ cao), không phải chuyện một **"chính đạo"** (người chơi bình thường) có thể làm được.

## Chạy thử

```bash
pip install -r requirements.txt

python -m src.tran_phap_3_chaos_prng.run
```

Tinh chỉnh:
```bash
# Quỹ đạo Lorenz dài hơn (đẹp hơn nhưng chậm hơn chút)
python -m src.tran_phap_3_chaos_prng.run --n-steps-attractor 15000

# Nhiều mẫu PRNG hơn để scatter 3D dày hơn
python -m src.tran_phap_3_chaos_prng.run --n-prng-samples 20000
```

Kết quả lưu tại `outputs/tran_phap_3/`:
- `hon_mang_lorenz.png` - trận đồ Lorenz + hiệu ứng cánh bướm
- `cau_truc_prng.png` - scatter 3D so sánh cấu trúc LCG yếu vs CSPRNG mạnh
- `tong_ket.json` - số liệu đầy đủ (số mũ Lyapunov, kết quả bắt mạch)

Với quy mô mặc định, toàn bộ pipeline chạy dưới **0.5 giây** trên máy CPU 8 luồng ~2.5GHz / 8GB RAM.

## Kỹ thuật đứng sau

1. **Tích phân RK4 (Runge-Kutta bậc 4)** tự viết cho hệ Lorenz - không phụ thuộc `scipy.integrate`, đủ chính xác cho mục đích minh họa.
2. **Ước lượng số mũ Lyapunov** bằng hồi quy tuyến tính `log(khoảng cách)` theo thời gian, chỉ trong đoạn còn "tuyến tính trên thang log" (trước khi khoảng cách bão hòa vì attractor có kích thước hữu hạn).
3. **Góc nhìn 3D được TÍNH TOÁN, không đoán mò**: với quan hệ đại số `x_{n+2} = 6*x_{n+1} - 9*x_n (mod 2^31)` của RANDU, pháp tuyến của họ mặt phẳng song song là `(9, -6, 1)`. Code tự tính một vector NẰM TRONG mặt phẳng đó (vuông góc pháp tuyến) để đặt góc camera - đây là lý do 15 dải phẳng hiện rõ thay vì chỉ là một khối chấm ngẫu nhiên như hầu hết các bài minh họa RANDU làm ẩu trên mạng.
4. **"Bắt mạch" chỉ với 1 số quan sát**: vì RANDU không có yếu tố bí mật nào ngoài trạng thái nội bộ, và trạng thái đó suy ra trực tiếp từ 1 giá trị đầu ra (`state = output × modulus`), nên biết 1 số là đủ để dựng lại toàn bộ tương lai của chuỗi.
