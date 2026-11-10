# Trận Pháp 5: Tâm Ma Suýt Trúng (Behavioral Agent Simulation)

> _"Tâm ma không nằm ở con số quay ra, mà ở chỗ ta diễn giải nó."_

## Câu hỏi

**Cảm giác "suýt trúng" có thể khiến một người ở lại cuộc chơi lâu hơn như thế nào?**

Bốn trận pháp trước nhìn vào kết quả và cách diễn giải xác suất. Trận pháp này chuyển sang
**quyết định của người chơi sau mỗi kết quả**. Quan trọng: suýt trúng **không** làm xác suất
kỳ sau tăng lên (các kỳ quay độc lập, có test kiểm chứng). Mô hình chỉ giả định agent *cảm nhận*
nó như một tín hiệu đáng hứng thú.

## Thí nghiệm

Mỗi agent mua **một vé mỗi kỳ** (6 số trong 1-55), nhận kết quả rồi quyết định tiếp tục hay dừng.
Dừng khi: tự dừng, không đủ tiền mua vé kỳ sau, hoặc chạm giới hạn số kỳ.

Mỗi agent được chạy **hai bản**, dùng CHUNG ngân sách, bộ số, chuỗi số trùng từng kỳ và cả số
ngẫu nhiên dùng để quyết định (common random numbers). Khác biệt duy nhất là β:

Các agent khác nhau có chuỗi kết quả độc lập. Đây là mô phỏng nhiều hành trình cá nhân,
không phải toàn bộ quần thể cùng tham gia một kỳ quay chung. Bộ số cụ thể chỉ dùng để minh họa
câu chuyện; động lực hiện tại phụ thuộc số lượng số trùng, được lấy mẫu trực tiếp từ phân phối siêu bội.

| Phiên bản | β | Cách phản ứng |
|---|---|---|
| Đối chứng | 0 | Hứng thú chỉ chịu tác động của kết quả tiền bạc |
| Nhạy suýt trúng | > 0 | Cùng quy tắc, thêm hứng thú khi gặp tín hiệu suýt trúng |

Hứng thú cập nhật theo

```
H(t+1) = rho * H(t) + alpha * R(t) + beta * S(t)        xác suất tiếp tục = sigmoid(H(t+1))
```

- `R(t)`: lãi/lỗ của kỳ, tính theo đơn vị giá vé, nén kiểu log (thua trọn vé = -0.69), chặn ở ±5.
- `S(t)`: điểm "suýt trúng" theo số trùng 3, 4, 5 (trùng 6 là jackpot, thắng thật, nên S=0).
- Tiền thưởng nằm trong `R`, cảm giác gần thắng nằm trong `S`: **ở những kỳ cả hai bản cùng chơi,
  tiền thưởng giống nhau**. Tổng tiền nhận có thể khác vì bản nhạy chơi thêm sau khi đối chứng dừng.

Bốn cách gán `S` để thử độ bền (`--cach-gan-diem`):

| Cách gán | trùng 3 | trùng 4 | trùng 5 |
|---|---|---|---|
| `tuyen_tinh` (mặc định) | 0.33 | 0.67 | 1 |
| `nhan_manh_gan_nhat` | 0.1 | 0.4 | 1 |
| `tu_3_so_nhu_nhau` | 1 | 1 | 1 |
| `chi_4_5_so` | 0 | 0.5 | 1 |

## Chạy thử

```bash
python -m src.tran_phap_5_near_miss.run
```

Tinh chỉnh:
```bash
python -m src.tran_phap_5_near_miss.run --beta 2.0 --n-agents 20000
python -m src.tran_phap_5_near_miss.run --cach-gan-diem tu_3_so_nhu_nhau --rho 0.995
python -m src.tran_phap_5_near_miss.run --no-sweep      # bỏ quét β và lưới độ bền cho nhanh
```

Kết quả tại `outputs/tran_phap_5/`: `tam_ma_suyt_trung.png` (tỷ lệ còn chơi, chi-nhận, câu chuyện
agent), `quet_beta_va_do_ben.png` (quét β, lưới độ bền), `tong_ket.json`.
10.000 agent x 2 bản chạy trong khoảng 2 giây.

## Kết quả tham chiếu (10.000 agent, β=1, `tuyen_tinh`, seed 42)

| | Đối chứng | Nhạy suýt trúng |
|---|---|---|
| Số kỳ trung bình | 18.77 | 19.28 (chênh +0.52, CI95 +0.46..+0.58) |
| Tiền chi trung bình | 187,656đ | 192,812đ |
| Có mua vé ở kỳ 20 | 43.7% | 45.0% |
| Có mua vé ở kỳ 30 | 16.0% | 17.7% |

Đường còn chơi tính `P(số kỳ đã chơi >= t)`: tỷ lệ có mua vé ở kỳ `t`, không phải tỷ lệ tiếp tục
sau khi hoàn tất kỳ đó. Ví dụ, 43.70% đối chứng mua vé kỳ 20; 40.49% tiếp tục sang kỳ 21.

- Kéo dài tương đối **+2.75%** ở seed 42; qua 10 seed khác nhau là **+2.3% đến +2.8%** (trung bình ~2.5%).
- Chỉ **5.1%** agent chơi lâu hơn; đó là nhóm từng gặp suýt trúng (21.4% quần thể). Nhóm chưa gặp
  thì hai bản trùng khít, chênh đúng 0.
- Quét β (kéo dài tương đối): β=0.5 → +1.2%, β=1 → +2.75%, β=2 → +5.6%, β=3 → +8.4%.

**Độ bền (β=1, % kéo dài tương đối):**

| Cách gán S \ ρ | 0.95 | 0.97 | 0.985 | 0.995 |
|---|---|---|---|---|
| `tuyen_tinh` | 1.5% | 2.0% | 2.8% | 3.3% |
| `nhan_manh_gan_nhat` | 0.5% | 0.7% | 0.9% | 1.3% |
| `tu_3_so_nhu_nhau` | 4.3% | 5.6% | 7.9% | 10.6% |
| `chi_4_5_so` | 0.1% | 0.2% | 0.2% | 0.4% |

Trong phạm vi tham số đã khảo sát, độ kéo dài thay đổi đáng kể giữa các cách gán điểm,
đặc biệt khi có hoặc không tính trùng 3 số. Trùng 3 số xảy ra ~1.27% mỗi kỳ, trùng 4-5 số chỉ
~0.06%. Các bảng điểm đổi cả loại sự kiện lẫn cường độ kích thích, nên chưa đủ cơ sở xếp hạng
mức nhạy với cách định nghĩa S so với mức nhạy với β: mô hình dùng tích `β*S`.

Thống kê "từng gặp suýt trúng" đếm kỳ có **S > 0 theo cấu hình**, chỉ trong các kỳ agent thực sự
chơi ở bản nhạy. Với `chi_4_5_so`, trùng 3 không được đếm. Đây là phân nhóm mô tả sau mô phỏng;
không dùng chênh lệch của nhóm này như một ước lượng tác động nhân quả giữa các nhóm người.

## Đọc kết quả thế nào cho đúng

1. **Chiều của hiệu ứng được lập trình sẵn.** Với cách ghép cặp này và β ≥ 0, bản nhạy không bao giờ
   chơi ít hơn bản đối chứng (test `test_ghep_cap_ban_nhay_khong_bao_gio_choi_it_hon`).
   Việc "nhóm nhạy suýt trúng chơi lâu hơn" **không phải một phát hiện**. Phần có thông tin là
   *độ lớn*, *điều kiện* và *độ bền khi đổi giả định*.
2. **Đây là phát hiện về mô hình, chưa phải về con người.** ρ, α, β, cách gán S, phân phối ngân sách,
   bảng thưởng đều là tham số giả định, chưa đo từ người thật.
3. Hiệu ứng nhỏ có lý do cấu trúc: với 1 vé/kỳ, suýt trúng hiếm, nên chỉ ~1/5 agent kịp gặp trong
   ~19 kỳ chơi.

## Giới hạn của bản đầu

- Chỉ 1 vé/kỳ; không có gambler's fallacy, không "quay lại chơi sau khi đã dừng".
- Bảng thưởng giả định (3 số 50.000đ, 4 số 500.000đ, 5 số 40 triệu, 6 số 30 tỷ; vé 10.000đ), jackpot
  cố định, không chia giải, không giải phụ; không phải bảng thưởng chính thức. Tiền thưởng kỳ vọng
  theo bảng này là ~23.8% giá vé, nhưng ~10 điểm % trong đó đến từ jackpot gần như không bao giờ
  xuất hiện trong mô phỏng hữu hạn.
- Suýt trúng chỉ định nghĩa qua số lượng số trùng. Lệch số ±1 để phiên bản sau (cần định nghĩa khoảng cách).
- Khoảng tin cậy chỉ phản ánh độ lệch giữa các agent trong một hạt giống; độ lệch giữa các seed
  xem ở trên.

Chi tiết kỹ thuật và cách chọn tham số: [`docs/tran_phap_5_ky_thuat.md`](../../docs/tran_phap_5_ky_thuat.md).
