# Trận Pháp 1: Vạn Kiếp Quy Tông (Monte Carlo Simulator)

> _"Chọn số đẹp không làm tăng Thiên Mệnh, nhưng sẽ khiến tiền thưởng bị phân tán như cát bụi."_

## Ý tưởng

Mô phỏng hai phe người chơi xổ số 6/55:

- **Phe Vô Vi** - chọn 6 số hoàn toàn ngẫu nhiên trong 1..55.
- **Phe Tình Cảm** - thiên vị chọn số ≤31 (ngày sinh) và vài số "hên"
  (`1, 9, 11, 20, 36`), dùng thuật toán weighted sampling.

**Sự thật toán học:** xác suất trúng của một vé bất kỳ luôn là `1 / C(55,6) ~ 1/28.989.675`, bất kể vé đó gồm số gì. Vì kết quả quay số  là đều (uniform) trên toàn bộ tổ hợp. 

Bí kíp không "chứng minh" điều này bằng cách chờ trúng thật (bất khả thi trong đời mô phỏng vài triệu vé), mà đo trực tiếp thứ quyết định "họa" thật sự: **mức độ các vé bị trùng nhau giữa những người chơi cùng phe**. Phe Tình Cảm dồn vào một vùng số hẹp hơn nhiều (chủ yếu 1..31 + vài số hên), nên xác suất hai người chọn trúng y hệt nhau cao hơn hẳn - nếu một trong các tổ hợp đó trúng jackpot, khả năng phải chia giải cho nhiều người cùng lúc tăng vọt.

## Chạy thử

Từ thư mục gốc repo (`khai-sang-thien-co/`):

### Một lần chạy thử (Single Trial)

Dùng để kiểm tra kết quả cơ bản

```bash
pip install -r requirements.txt

# Mặc định: 2 triệu vé / phe - hợp với máy 8GB RAM, chạy vài chục giây
python -m src.tran_phap_1_monte_carlo.run

# Máy yếu hơn / muốn chạy nhanh để test: giảm quy mô
python -m src.tran_phap_1_monte_carlo.run --n-players 300000 --chunk-size 50000

# Máy khỏe, muốn xem rõ hơn: tăng quy mô (chú ý RAM)
python -m src.tran_phap_1_monte_carlo.run --n-players 10000000 --chunk-size 500000

# Output được lưu vào: outputs/tran_phap_1_YYYYMMDD_HHMMSS/
```

Kết quả lưu tại `outputs/tran_phap_1/`:
- `so_sanh_hai_phe.png` - biểu đồ so sánh 2 phe
- `top_to_hop_pho_bien_tinh_cam.csv` - 15 tổ hợp được Phe Tình Cảm chọn nhiều nhất
- `tong_ket.json` - số liệu tổng kết dạng máy đọc được

**Lưu ý:** Mỗi lần chạy sẽ tạo một thư mục mới với timestamp để tránh ghi đè. Nếu muốn ghi đè lên thư mục cũ, thêm --no-timestamp.

### Tu tiên lă lại (Repeated Trials)

Dùng để chạy nhiều lần độc lập, gộp nhiê kết quả va 1 báo cáo tổng hợp. Cách này giúp nhìn rõ chân lý nhất

```
# Mặc định: 100 lần × 5 triệu vé/phe/lần = 500 triệu vé/phe
python -m src.tran_phap_1_monte_carlo.run_repeated

# Chạy ít hơn để test nhanh
python -m src.tran_phap_1_monte_carlo.run_repeated \
    --n-players-per-run 500000 \
    --n-runs 10

# Chạy thêm 50 lần nữa (tự động nối vào CSV cũ)
python -m src.tran_phap_1_monte_carlo.run_repeated --n-runs 50

# Bắt đầu lại từ đầu (xóa dữ liệu cũ)
python -m src.tran_phap_1_monte_carlo.run_repeated --fresh

# Chỉ tổng hợp dữ liệu đã có (không chạy sim mới)
python -m src.tran_phap_1_monte_carlo.run_repeated --aggregate-only

# Output được lưu vào: outputs/tran_phap_1_repeated_YYYYMMDD_HHMMSS/
```

Sản phẩm sau khi chạy `run_repeated`:
`ket_qua_theo_lan_chay.csv`:	    Chi tiết từng lần chạy (mỗi dòng 1 lần)
`tong_hop_nhieu_lan_chay.json`:	Tổng kết: trung bình, độ lệch chuẩn, min, max
`tong_hop_nhieu_lan_chay.png`:	Biểu đồ: đường biến thiên + hộp phân phối

### Cấu hình theo máy (cho `run_repeated.py`)

| Cấu hình | Khuyến nghị | Lý do |
|----------|-------------|-------|
| **Máy yếu (4GB RAM)** | `--n-players-per-run 500000 --n-runs 50` | Tránh quá tải |
| **Máy trung bình (8GB RAM)** | `--n-players-per-run **5.000.000** --n-runs 100` | **Con số vàng!** Đủ lớn để thấy chân lý, vẫn chạy mượt |
| **Máy khỏe (16GB+ RAM)** | `--n-players-per-run 10.000.000 --n-runs 100` | Muốn "vĩ đại" hơn nữa |

**Lưu ý:** 8GB RAM của đạo hữu có thể chạy tới 20 triệu vé/lần nếu chỉ chạy 1 lần (dùng `run.py`), nhưng với `run_repeated.py`, 5 triệu/lần là tối ưu để chạy 100 lần mà không lo máy "tắt thở".

### So sánh nhanh 2 cách chạy

| | `run.py`| `run_repeated.py`|
|--|--------|-------------------|
|Mục đích | Kiểm tra nhanh 1 lần | Chứng minh chân lý bằng thống kê |
|Số vé mỗi lần chạy | 2 triệu (mặc định) | 5 triệu (mặc định) | 
|Số lần chạy | 1 lần | 100 lần (mặc định) |
|Tổng vé mô phỏng | 2 triệu/phe | 500 triệu/phe |
|Thời gian | ~30-60 giây | ~60 phút |
|Output | 3 file | 3 file (nhưng có ý nghĩa về thống kê) |

## Vì sao có cả `run.py` và `run_repeated.py

- run.py sinh ra từ thời kỳ "sơ khai" của bí kíp, dùng để chạy thử nhanh một lần.
- run_repeated.py là phiên bản "tu tiên" hiện đại, chạy nhiều lần để hội tụ thống kê, cho kết quả chính xác và ổn định hơn.

Khuyến nghị dùng `run_repeated.py` nếu đạo hữu muốn thấy "chân lý" rõ ràng. run.py chỉ để test nhanh hoặc chạy khi không có nhiều thời gian.

## Vì sao không sinh "1 tỷ vé" như bí kíp gốc đòi hỏi ?

Vì máy tính có hạn (đặc biệt trên laptop CPU/RAM khiêm tốn), 1 tỷ vé × 2 phe sẽ tốn nhiều phút đến hàng giờ và có thể vượt RAM nếu code viết ẩu.

- run.py dùng 2 triệu vé/phe - đã đủ để hiện tượng "trùng vé" giữa Phe Tình Cảm xuất hiện rõ rệt và có ý nghĩa thống kê.
- run_repeated.py dùng 100 lần × 5 triệu vé/phe = 500 triệu vé/phe - tương đương gần 1 tỷ lần "ảo" mà không tốn SSD.

Code đã được vector hóa + chia lô (chunk) nên tăng tuyến tính theo thời gian chứ không nổ RAM.

### Giải thích thêm về "1 tỷ ảo"

> _10 triệu vé x 100 lần = 1 tỷ vé mô phỏng_

Nhưng với cấu hình khuyến nghị (`5 triệu x 100 = 500 triệu vé`), vẫn là một con số cực kỳ lớn để thấy chân lý rõ ràng. Nêu muốn đạt 1 tỷ, đạo hư có thể lên thành `10 triệu x 100` nhưng sẽ mất tận 2 giờ chạy mô phỏng.

## Kỹ thuật đứng sau (cho đạo hữu tò mò)

1. **Sinh vé không lặp số, có trọng số, vector hóa 100%** bằng thủ pháp Efraimidis-Spirakis: mỗi số được gán key = U^(1/w), lấy 6 số có key lớn nhất mỗi hàng. Làm được cho hàng triệu vé cùng lúc bằng một phép toán ma trận numpy duy nhất - không có for-loop qua từng vé.
2. **Mã hóa vé thành 1 số nguyên** (hệ cơ số 56) để đếm trùng lặp bằng np.unique trong một lần quét, thay vì so sánh từng cặp vé (sẽ là O(n²), giết chết máy yếu ngay cả với vài chục nghìn vé).
3. **Chia lô (chunking):** sinh vé theo từng đợt, giải phóng ma trận trung gian ngay sau khi mã hóa - giữ bộ nhớ đỉnh chỉ khoảng vài chục MB bất kể tổng số vé muốn sinh là bao nhiêu.

## Cấu trúc thư mục output

``` text
outputs/
├── tran_phap_1_20260101_210315/         # Từ run.py (có timestamp)
│   ├── so_sanh_hai_phe.png
│   ├── top_to_hop_pho_bien_tinh_cam.csv
│   └── tong_ket.json
│
├── tran_phap_1_20260101_215542/         # Lần chạy khác của run.py
│   └── ...
│
└── tran_phap_1_repeated_20260101_230000/  # Từ run_repeated.py
    ├── ket_qua_theo_lan_chay.csv
    ├── tong_hop_nhieu_lan_chay.json
    └── tong_hop_nhieu_lan_chay.png
```

Không còn nỗi lo "tàn canh gió lạnh" vì ghi đè kỹ thuật

---

**Chúc đạo hữu tu tiên thành công, ngộ ra chân lý của vũ trụ xác suất!**

> _Chọn số đẹp không làm tăng Thiên Mệnh, nhưng sẽ khiến tiền thưởng bị phân tán như cát bụi._