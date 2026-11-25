# Công cụ tái lập bài báo

## Baseline tần suất gần đây bổ sung cho RQ2

```bash
python paper/experiments/run_recent_frequency.py
python paper/experiments/make_figures.py --heuristic-data outputs/paper_recent_frequency_v1/B_runs.csv
```

Heuristic chọn sáu số xuất hiện nhiều nhất trong 52 kỳ trước mỗi mục tiêu test,
phá hòa bằng số tăng dần. Dùng lại seeds 0–29 và seed 42 riêng, cùng 151 mục
tiêu test của B; không fit và không gán confidence score. Đây là bổ sung được
thiết kế sau khi xem kết quả cây, không phải thực nghiệm đăng ký trước.

Kết quả riêng ở `outputs/paper_recent_frequency_v1/`: mean overlap
0.6620309051, sample SD 0.0621364770, range 0.5496688742–0.8211920530;
seed 42 là 0.6225165563 và không tính vào aggregate. Kỳ vọng giải tích là
36/55. Script lưu lựa chọn từng kỳ, CSV theo seed, aggregate, source snapshot
và manifest có phiên bản môi trường cùng SHA256. Không sửa frozen v1 và từ
chối ghi đè thư mục đã tồn tại; có thể truyền `--output-dir` mới để tái lập.

Kiểm tra chống rò rỉ và phá hòa:
`python -m pytest -q tests/test_recent_frequency.py tests/test_tran_phap_2.py`.

Thư mục này chỉ chứa mã Python, cấu hình và dữ liệu xuất cho bài báo.
Các lệnh dưới đây chạy từ gốc repository; đường dẫn mặc định của script được
xác định theo vị trí file nên cũng có thể gọi bằng đường dẫn tuyệt đối từ nơi khác.
Không chứa bản build Overleaf hoặc LaTeX.

## Kiểm tra bộ kết quả đã đóng băng

```bash
python paper/experiments/run_frozen.py --verify-only
```

Kiểm tra SHA256 của nguồn/cấu hình trong `source_snapshot/`, các kết quả và
đầu vào lịch sử theo `outputs/paper_frozen_v1/manifest.json`. Lệnh này không
khẳng định mã đang làm việc giống bản lịch sử. Nếu cần so khớp cả đường dẫn
và nội dung mã hiện tại, thêm `--check-working-tree`; bản đã refactor sẽ
không khớp bản nguồn cũ và chế độ đó sẽ báo lỗi.

Ghi chú kiểm tra ngày 25/11/2026: checkout hiện tại có 20 mục lệch checksum
do chuyển kiểu xuống dòng LF/CRLF, nên `--verify-only` chưa đạt kiểm tra byte
nguyên gốc. Đối chiếu 113 mục nguồn snapshot/kết quả: cả 20 mục đều khớp
checksum lưu trữ khi đổi lại kiểu xuống dòng, không phát hiện lệch nội dung
khác. Các file và manifest frozen được giữ nguyên; không diễn giải kiểm tra
chuẩn hóa xuống dòng thành việc kiểm tra byte nguyên gốc đã đạt.

## Chạy lại thực nghiệm

```bash
python paper/experiments/run_frozen.py --output-dir outputs/paper_reproduced_v1
```

Audit dữ liệu A lịch sử, chạy B/C theo cấu hình cố định và lưu manifest cùng
source snapshot mới. Script từ chối ghi đè thư mục kết quả đã tồn tại.
Giữ nguyên `frozen_config.json` và các đường dẫn dữ liệu lịch sử để bảo toàn
nguồn gốc thực nghiệm. Bản gốc `outputs/paper_frozen_v1/` không bị sửa.

## Xuất hình

```bash
python paper/experiments/make_figures.py
```

Xuất ba hình, mỗi hình có PDF vector và PNG 300 dpi, vào
`outputs/paper_figures/` (được Git bỏ qua). Có thể chỉ định `--config`, `--data`
và `--out`. Các PDF thiết kế cho bố cục hai cột, bề rộng khoảng 7,1 inch.

## Tái xuất dữ liệu và audit phân nhóm

```bash
python paper/experiments/verify_and_export.py
python paper/experiments/audit_near_miss_groups.py
```

`verify_and_export.py` tính lại A/B/C, đối chiếu các số đã ghi trong script
và ghi `A_summary.json`, `B_runs.csv`, `C_runs.csv`, `C_ref.json` vào
`paper_data/`. Dùng `--out` để xuất sang thư mục khác hoặc `--skip A B C`
để bỏ qua các thực nghiệm tương ứng.

`audit_near_miss_groups.py` kiểm tra các dữ liệu hình với kết quả frozen,
tái tạo mười quần thể C rồi ghi `paper_data/C_subgroups.json`.

Phân biệt các nhóm mô tả sau mô phỏng:

- Từng gặp near-miss: 21,059% agent; tăng trung bình 2,22812 vòng trong nhóm.
- Chơi thêm ít nhất một vòng: 4,739% agent; tăng trung bình 9,90124 vòng trong nhóm.
- Nhóm chơi thêm: control trung bình 24,70563 và sensitive 34,60688 vòng,
  tương đương tăng 40,07687%.

Đây là mô tả phân nhóm sau mô phỏng, không phải ước lượng nhân quả phân nhóm.
