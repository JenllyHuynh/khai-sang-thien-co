# Chỉ mục kết quả mô phỏng

| Nhóm | Thư mục | Vai trò |
| --- | --- | --- |
| **Paper evidence (frozen)** | [paper_frozen_v1/](paper_frozen_v1/), [paper_recent_frequency_v1/](paper_recent_frequency_v1/) | Kết quả chính của bài báo và bộ bổ sung Experiment B. Giữ nguyên vị trí, nội dung và manifest. |
| **Archived reference inputs (used by paper audit)** | [tran_phap_1_repeated_20260824_223348/](tran_phap_1_repeated_20260824_223348/), [tran_phap_1_20260824_222122/](tran_phap_1_20260824_222122/), [tran_phap_2_20260825_110043/](tran_phap_2_20260825_110043/), [tran_phap_5/](tran_phap_5/) | Nguồn 100 runs và ví dụ 2 triệu vé của Exp A; tham chiếu seed 42 lịch sử của Exp B và C. Giữ nguyên vị trí và nội dung để các đường dẫn audit tiếp tục hoạt động. |
| **Exploratory runs** | [exploratory/](exploratory/) | Các lần chạy thử, tách khỏi bằng chứng của bài báo. |

`exploratory/` giữ các bản sau:

- `tran_phap_3_20260825_110810/`: bản mới nhất của Trận pháp 3 ngày 25/08/2026.
- `tran_phap_4_20260825_111527/`: bản mới nhất của Trận pháp 4 ngày 25/08/2026.
- `tran_phap_5_20261109_155916/`: bản chạy thử beta = 2.
- `tran_phap_5_chi_4_5_da_sua/`: bản chạy thử quy tắc 4–5 số.

Đợt dọn ngày 25/11/2026 đã bỏ hai bản cũ của mỗi Trận pháp 3 và 4.

Hình bài báo nằm riêng tại [paper/figures/](../paper/figures/), gồm PDF và PNG;
được chuyển từ `outputs/paper_figures_recent_frequency/`. Đây là hình xuất từ
dữ liệu, không thuộc checksum trong hai manifest của nhóm paper evidence.
Để xuất lại hình có baseline recent-frequency, chạy từ gốc repository:

```bash
python paper/experiments/make_figures.py --heuristic-data outputs/paper_recent_frequency_v1/B_runs.csv
```
