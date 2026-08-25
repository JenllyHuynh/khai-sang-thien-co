# Tài Liệu Kỹ Thuật - Trận Pháp 3: Phá Giải Hỗn Mang (Chaos vs. PRNG)

## 1. Bài toán cần giải

Tách bạch 2 khái niệm hay bị nhầm lẫn:

- **"Trông hỗn loạn"** (chaos) - một hệ TẤT ĐỊNH nhưng cực nhạy với điều kiện ban đầu, khiến quỹ đạo trông như ngẫu nhiên dù công thức hoàn toàn xác định.
- **"Thực sự khó đoán"** (cryptographically unpredictable) - một chuỗi số mà không có thuật toán khả thi nào (trong thời gian hợp lý) có thể dự đoán được số tiếp theo dù biết toàn bộ lịch sử trước đó.

Trận pháp chứng minh: không phải cứ "trông rối rắm" là an toàn để làm bộ sinh số cho xổ số - cả 2 phần của trận pháp (Lorenz và LCG yếu) đều là hệ TẤT ĐỊNH, nhưng chỉ CSPRNG mạnh mới đạt được độ khó đoán thực sự.

## 2. Phần 1: Hệ Lorenz - Hỗn Loạn Tất Định

### 2.1. Phương trình

Hệ 3 phương trình vi phân thường (Lorenz, 1963), mô hình hóa đối lưu khí quyển đơn giản hóa:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

Tham số cổ điển (dùng trong code): `σ = 10`, `ρ = 28`, `β = 8/3` - đây là bộ tham số nổi tiếng tạo ra "bướm Lorenz" (Lorenz butterfly), một trong những attractor lạ (strange attractor) đầu tiên được phát hiện.

### 2.2. Tích phân số bằng Runge-Kutta bậc 4 (RK4)

Hệ phương trình vi phân không có nghiệm dạng đóng (closed-form), phải giải bằng phương pháp số. RK4 là lựa chọn cân bằng giữa độ chính xác và chi phí tính toán (sai số cục bộ O(dt⁵), sai số toàn cục O(dt^4)):

```python
k1 = f(state)
k2 = f(state + dt/2 * k1)
k3 = f(state + dt/2 * k2)
k4 = f(state + dt * k3)
state_next = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

trong đó `f(state) = (dx/dt, dy/dt, dz/dt)` (hàm `_lorenz_deriv`). Code tự cài đặt RK4 (không phụ thuộc `scipy.integrate`) - đủ chính xác cho mục đích minh họa và giữ dependency tối giản.

**Độ phức tạp:** O(n_steps) - mỗi bước chỉ cần 4 lần gọi hàm đạo hàm (mỗi lần là phép toán O(1) trên vector 3 chiều).

### 2.3. Hiệu ứng cánh bướm - đo định lượng bằng số mũ Lyapunov

**Thí nghiệm:** chạy 2 quỹ đạo với điều kiện ban đầu sai khác cực nhỏ:

```python
state_a = (1.0, 1.0, 1.0)
state_b = state_a + (1e-8, 0, 0)      # nhiễu 1e-8 chỉ ở trục x
distance(t) = ||trajectory_a(t) - trajectory_b(t)||₂
```

**Lý thuyết:** với một hệ hỗn loạn, khoảng cách giữa 2 quỹ đạo gần nhau tăng theo hàm mũ trong giai đoạn đầu (trước khi bão hòa vì attractor có kích thước hữu hạn):

```
distance(t) ~ distance(0) × e^(λt)
```

trong đó `λ` là **số mũ Lyapunov lớn nhất** - giá trị tham khảo lý thuyết cho hệ Lorenz cổ điển là `λ ~ 0.9056`.

**Ước lượng thực nghiệm (`stats.py`):** lấy log 2 vế:

```
log(distance(t)) ~ log(distance(0)) + λt
```

-> đây là một đường thẳng trên đồ thị `log(distance)` theo `t`, với hệ số góc chính là `λ`. Code dùng hồi quy tuyến tính (`np.polyfit`, bậc 1) trên đoạn `distance < distance_ceiling` (mặc định 1.0) - chỉ lấy đoạn TRƯỚC khi khoảng cách bão hòa (vì sau đó quan hệ không còn tuyến tính trên thang log, do quỹ đạo bị giới hạn trong kích thước hữu hạn của attractor).

**Thời gian nhân đôi sai số:**

```
doubling_time = ln(2) / λ
```

Đo thực tế: sai số `1e-8` nhân đôi mỗi ~1.17 đơn vị thời gian, số mũ Lyapunov ước lượng ~0.59-0.9 tùy quy mô mô phỏng (dao động do đây là ước lượng thống kê từ dữ liệu hữu hạn, không phải giá trị giải tích).

## 3. Phần 2: LCG Yếu (RANDU) vs CSPRNG Mạnh (PCG64)

### 3.1. Linear Congruential Generator (LCG)

Công thức tổng quát:

```
x_{n+1} = (a × x_n + c) mod m
```

**RANDU** (IBM, 1968) dùng `a = 65539 = 2^16 + 3`, `c = 0`, `m = 2^31` - lựa chọn tham số này (đặc biệt `a = 2^16 + 3`) tạo ra một khiếm khuyết toán học nghiêm trọng được Marsaglia phát hiện năm 1968.

### 3.2. Khiếm khuyết "siêu phẳng" (Marsaglia 1968)

**Định lý (Marsaglia):** với LCG dạng `x_{n+1} = a*x_n mod m`, luôn tồn tại một quan hệ TUYẾN TÍNH giữa `k` số hạng liên tiếp bất kỳ (`k` phụ thuộc kích thước không gian và `a`), do bản chất modular arithmetic của phép sinh số. Với RANDU cụ thể, quan hệ đó là:

```
x_{n+2} = 6*x_{n+1} - 9*x_n   (mod 2^31)
```

Kiểm chứng trực tiếp trong code (đã test - xem `tests/test_tran_phap_3.py` và log phát triển): với mọi bộ ba liên tiếp, `(x_{n+2} - 6x_{n+1} + 9x_n) mod 2^31 = 0` - đúng tuyệt đối, không có ngoại lệ.

**Diễn dịch hình học:** nếu coi mỗi bộ ba `(x_n, x_{n+1}, x_{n+2})` là một điểm trong không gian 3D `[0,1)^3` (sau khi chuẩn hóa `/m`), thì MỌI điểm đều nằm trên MỘT trong số ít các **mặt phẳng song song** có phương trình:

```
9*x_n - 6*x_{n+1} + x_{n+2} = k     (k nguyên)
```

Với biên độ tọa độ `[0,1)`, `k` chỉ nhận khoảng **15 giá trị nguyên khác nhau** - nghĩa là toàn bộ không gian 3D chỉ có 15 mặt phẳng chứa TẤT CẢ các điểm dữ liệu, thay vì lấp đầy không gian một cách "ngẫu nhiên".

### 3.3. Góc nhìn 3D được TÍNH TOÁN, không đoán mò

**Vấn đề:** nhìn từ một góc bất kỳ, 15 mặt phẳng chồng lấn/xen kẽ trông không khác gì một khối mây điểm ngẫu nhiên - cấu trúc chỉ lộ ra khi nhìn ĐÚNG GÓC.

**Giải pháp toán học:** pháp tuyến của họ mặt phẳng song song (từ phương trình `9x - 6y + z = k`) là vector:

```
n = (9, -6, 1)
```

Để THẤY các mặt phẳng dưới dạng "lát mỏng xếp chồng" (không nhìn thẳng diện vào mặt phẳng, mà nhìn dọc theo bề mặt của nó), camera phải hướng nhìn dọc theo một vector NẰM TRONG mặt phẳng - tức VUÔNG GÓC với pháp tuyến `n`. Code chọn:

```python
v = (6, 9, 0)              # kiểm tra: n*v = 9×6 + (-6)×9 + 1×0 = 0 (vuông góc)
azim = degrees(atan2(v_y, v_x))    # góc quay quanh trục z
elev = 8.0                          # nghiêng nhẹ để có chiều sâu 3D
```

**Kết quả thực nghiệm:** với góc này, scatter 3D của LCG yếu hiện rõ **15 dải song song rành mạch** cùng góc nhìn áp lên dữ liệu từ CSPRNG mạnh chỉ cho ra một khối mây đặc, không cấu trúc - tương phản trực quan mạnh mẽ mà không cần "tự thử" hàng chục góc bằng mắt.

### 3.4. "Bắt mạch" - State Reconstruction Attack

**Nguyên lý:** RANDU không có yếu tố bí mật nào ngoài trạng thái nội bộ `x_n` - công thức `a, c, m` đều công khai. Vì phép chia `output = state / m` là một song ánh (bijection) khi biết `m`, chỉ cần **1 giá trị đầu ra quan sát được** là suy ngược lại chính xác `state`:

```python
state = round(observed_output × m)
```

Từ đó, áp dụng công thức LCG để dự đoán TOÀN BỘ các giá trị tiếp theo với sai số **tuyệt đối = 0** (không phải "gần đúng". Vì mọi phép toán đều là số nguyên modular, không có sai số làm tròn tích lũy đáng kể trong phạm vi `float64`).

```python
def attempt_state_reconstruction_crack(observed):
    state = round(observed[0] × MODULUS) % MODULUS
    predicted = [LCG(state).next_float() for _ in range(len(observed)-1)]
    error = max(|predicted - observed[1:]|)
```

Đo thực tế: `max_abs_error = 0.00e+00` - crack tuyệt đối chỉ với 1 số.

**Đối chứng:** áp dụng ĐÚNG kỹ thuật này lên dữ liệu từ CSPRNG mạnh (PCG64) - vì PCG64 không tuân theo công thức LCG tuyến tính đơn giản này, phép "bắt mạch" thất bại ngay từ số dự đoán đầu tiên, sai số `~0.5-1.0` (tương đương đoán ngẫu nhiên hoàn toàn).

### 3.5. Ý nghĩa thực tiễn

Đây chính là lý do các cao thủ Cryptography xem việc "bắt mạch" LCG hay Mersenne Twister (thuật toán tương tự, cần quan sát 624 giá trị liên tiếp để dựng lại toàn bộ trạng thái nội bộ) là chuyện nhỏ và cũng là lý do xổ số điện tử thật KHÔNG BAO GIỜ dùng các PRNG này, mà dùng **CSPRNG** (Cryptographically Secure PRNG) hoặc **TRNG** (True RNG, đo nhiễu vật lý như nhiệt/lượng tử).

## 4. Độ phức tạp

| Bước | Độ phức tạp |
|--|-|
| Tích phân RK4 (trận đồ Lorenz) | O(n_steps) |
| Hiệu ứng cánh bướm (2 quỹ đạo) | O(n_steps) |
| Ước lượng số mũ Lyapunov | O(n_steps) (hồi quy tuyến tính) |
| Sinh LCG/CSPRNG | O(n_samples) |
| Bắt mạch state reconstruction | O(n_samples) |

Đo thực tế: toàn bộ pipeline (6000 bước Lorenz + 3000 bước cánh bướm +
5000 mẫu mỗi PRNG) chạy **dưới 0.5 giây** trên CPU 8 luồng ~2.5GHz.

## 5. Tham chiếu

- Lorenz, E. N. (1963). *Deterministic Nonperiodic Flow*. Journal of the Atmospheric Sciences, 20(2), 130-141.
- Marsaglia, G. (1968). *Random numbers fall mainly in the planes*. Proceedings of the National Academy of Sciences, 61(1), 25-28.
- L'Ecuyer, P. (2017). *History of Uniform Random Number Generation* - tổng quan về LCG, Mersenne Twister, và các CSPRNG hiện đại (PCG, ChaCha20...).
- numpy `default_rng()` sử dụng **PCG64** làm bit generator mặc định - vượt qua bộ kiểm định TestU01 BigCrush và tương tự NIST SP 800-22.
