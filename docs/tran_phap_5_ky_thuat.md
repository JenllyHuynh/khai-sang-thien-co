# Tài Liệu Kỹ Thuật - Trận Pháp 5: Tâm Ma Suýt Trúng (Behavioral Agent Simulation)

## 1. Bài toán

Câu hỏi: *giả sử* người chơi phản ứng với tín hiệu "suýt trúng" bằng một lượng hứng thú cộng thêm,
thì cơ chế giả định đó kéo dài thời gian chơi bao nhiêu, trong điều kiện nào, và kết quả còn giữ khi đổi tham số không?

Đây là câu hỏi về **hệ quả của một giả định**, không phải kiểm định một giả thuyết về người thật.
Nhóm "nhạy suýt trúng" chơi lâu hơn là hệ quả do thiết kế; con số cần đọc là độ lớn và độ nhạy.

## 2. Kiến trúc

```
config.py      ->  luật chơi 6/55, giá vé, bảng thưởng, các cách gán điểm S, ThamSo
draws.py       ->  phân phối số trùng (siêu bội), quay số thật, dựng kỳ quay có k số trùng
agents.py      ->  quần thể: ngân sách, bộ 6 số, H0
dynamics.py    ->  vòng lặp theo kỳ, vector hóa trên agent (chay_mot_ban)
simulate.py    ->  hạt giống chung, chạy cặp, quét beta, lưới độ bền
stats.py       ->  so sánh ghép cặp, bootstrap, đường còn chơi
story.py       ->  chọn agent đại diện, dựng câu chuyện
visualize.py   ->  biểu đồ chính + biểu đồ quét/độ bền
run.py         ->  CLI
```

## 3. Mô hình agent

Mỗi agent: ngân sách `B` (số vé ~ lognormal, trung vị 40, sigma 0.8, nhân giá vé 10.000đ), bộ 6 số
(đều ngẫu nhiên trong 1-55), hứng thú ban đầu `H0 ~ N(4.5, 0.8)`.

Thứ tự trong mỗi kỳ `t` cho agent đang chơi:

1. Mua 1 vé (trừ giá vé), nhận thưởng theo số trùng.
2. `x = (thưởng - giá vé) / giá vé`; `R = clip(sign(x) * ln(1 + |x|), -5, 5)`.
   Thua trọn vé: R = -0.69; trùng 3: x=4 → R=1.61; trùng 4: x=49 → R=3.93; trùng 5 trở lên chạm trần 5.
3. `S = bảng điểm[số trùng]` (0 nếu trùng 0-2 hoặc 6).
4. `H <- rho*H + alpha*R + beta*S`.
5. Nếu tiền còn lại < giá vé → dừng (hết ngân sách). Nếu không, tiếp tục với xác suất `sigmoid(H)`, ngược lại tự dừng.
   Hết ngân sách được xét trước tự dừng. Dừng là vĩnh viễn.

Chặn `|R| ≤ 5` để vài kỳ trúng lớn cực hiếm không áp đảo động lực (hai bản vẫn nhận tiền như nhau).

## 4. Hai bản ghép cặp

`tao_hat_giong` sinh một lần, cho mỗi agent: ngân sách, bộ số, H0, ma trận số trùng `M[i,t]`, ma trận
`U[i,t]` cho quyết định tiếp tục (so với `sigmoid(H)`). Cả hai bản dùng cùng các ma trận này
(common random numbers), nên chênh lệch giữa hai bản chỉ đến từ `beta*S`.

**Hệ quả toán học:** khi cả hai bản còn chơi thì ngân sách và `R` giống hệt nhau; `S ≥ 0`, `beta ≥ 0`
nên `H_nhạy ≥ H_đối chứng` ở mọi kỳ; vì `U` chung, nếu bản đối chứng tiếp tục thì bản nhạy cũng tiếp tục.
Vậy:

- Mỗi agent: số kỳ bản nhạy ≥ số kỳ bản đối chứng (không có ngoại lệ).
- Agent chưa gặp kỳ nào `S > 0` khi còn chơi: hai bản trùng khít.
- `beta = 0` cho hai bản giống hệt nhau (kiểm tra tính đúng của cách cài đặt).
- `beta` lớn hơn không bao giờ làm agent chơi ít đi.

Tất cả đã có test. Đổi lại, vì chiều hiệu ứng là bắt buộc, kết luận định tính "chơi lâu hơn" không mang
thông tin; báo cáo tập trung vào độ lớn.

### Vì sao lấy số trùng trực tiếp thay vì quay 55 số

Với vé cố định, số trùng ~ siêu bội(55, 6, 6) bất kể vé là gì. `sample_so_trung` lấy thẳng phân phối đó
(nhanh hơn nhiều với hàng triệu cặp agent-kỳ). Test đối chiếu với quay 6/55 thật. `dung_ky_quay_co_k_trung`
dựng một kỳ quay có đúng k số trùng (chọn k số từ vé, 6-k số ngoài vé) chỉ để hiển thị câu chuyện.

## 5. Điểm "suýt trúng"

Chỉ dùng số lượng số trùng (3, 4, 5). Bốn cách gán trong `config.CACH_GAN_DIEM`. Agent *cảm nhận* kỳ
sát nút là điềm gần jackpot, nhưng các kỳ quay độc lập: test
`test_cac_ky_quay_doc_lap_suyt_trung_khong_bao_truoc_ky_sau` kiểm tra xác suất suýt trúng kỳ sau không đổi
sau một lần suýt trúng.

Trùng 3-5 số cũng có thưởng. Vì vậy `R` (tiền) và `S` (cảm giác) tách riêng; đối chứng vẫn được kích thích
bởi tiền thưởng, và bản nhạy chỉ có thêm phần `beta*S`.

## 6. Thống kê

- So sánh ghép cặp theo từng agent: chênh số kỳ, tiền chi, thua ròng.
- Khoảng tin cậy 95% bootstrap (percentile) theo agent, resample chia khối để tiết kiệm RAM.
- Đường còn chơi: `S(t) = P(số kỳ ≥ t)` cho mỗi bản. Số kỳ trung bình là trung bình bị chặn ở `max_ky`
  (mặc định 300; thực tế gần như không agent nào chạm).
- Nhóm đã/chưa từng gặp suýt trúng được báo riêng để thấy hiệu ứng tập trung ở ai.
- Quét β và lưới độ bền (cách gán S x ρ) dùng chung quần thể và chuỗi quay; mỗi ô lưới có đối chứng riêng
  vì ρ làm đổi hành vi của chính nhóm đối chứng.

Lưu ý: bootstrap chỉ đo độ lệch giữa các agent trong một seed. Qua 10 seed (N=10.000, β=1) kéo dài tương
đối dao động từ +2.3% đến +2.8% (trung bình ~2.5%, độ lệch chuẩn ~0.15 điểm %).

## 7. Cách chọn tham số mặc định

Tham số mặc định được chọn theo tiêu chí về **nhóm đối chứng**, không chọn để hiệu ứng trông lớn:

- Lần thử đầu (ρ=0.95, α=0.25, H0~3): agent đối chứng bỏ cuộc sau khoảng 6 kỳ, hầu như chưa kịp gặp
  suýt trúng (xác suất trùng 3-5 số chỉ ~1.3%/kỳ), nên hiệu ứng ~0.
- Trong vùng ρ=0.97-0.99 với α rất nhỏ, độ dài chơi gần như không phụ thuộc α, tức kết quả tiền bạc bị
  vô hiệu hóa. Loại.
- Mặc định (ρ=0.985, α=0.06, H0~N(4.5, 0.8)): trung vị ~18-19 kỳ, tiền bạc có tác động thật,
  ~80% tự dừng, ~20% hết ngân sách. Cách chọn này vẫn là giả định và phải đọc như vậy.

Công thức không có hằng số nền nên ở trạng thái dừng (chỉ thua) `H` luôn âm; thời gian chơi do giai đoạn
chuyển tiếp từ `H0` quyết định. Đó là một giới hạn của dạng công thức, ghi lại để khỏi hiểu nhầm.

## 8. Giới hạn và hướng mở rộng

- Nhiều vé mỗi kỳ (tăng số lần gặp suýt trúng, có thể tăng hiệu ứng).
- Suýt trúng kiểu lệch số ±1 (cần định nghĩa khoảng cách giữa các số).
- Hiệu chỉnh ρ, α, β bằng dữ liệu hành vi thật; không có bước này thì không rút ra được kết luận về người chơi thật.
- Mô hình dừng không quay lại, không có gambler's fallacy, không tương tác giữa các agent.
- Bảng thưởng là giả định minh họa, cần thay bằng luật thật nếu muốn bám một loại xổ số cụ thể.
