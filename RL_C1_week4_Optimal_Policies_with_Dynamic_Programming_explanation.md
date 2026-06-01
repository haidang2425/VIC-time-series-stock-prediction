# Giải thích chi tiết: RL_C1_week4_Optimal_Policies_with_Dynamic_Programming.ipynb

## Mục tiêu chung
Bài tập này mô phỏng một bài toán chính sách tối ưu (optimal policy) với quy hoạch động (dynamic programming) trong một môi trường quản lý bãi đỗ xe.

- Mục tiêu chính: tìm chính sách giá tốt nhất cho bãi đỗ xe sao cho tối đa hóa giá trị kỳ vọng dài hạn.
- Bài toán sử dụng mô hình MDP (Markov Decision Process).
- Ta làm việc với 3 thuật toán giáo trình chính: policy evaluation, policy iteration và value iteration.

---

## Yêu cầu đề bài

Bài tập yêu cầu bạn hoàn thành các phần sau:

1. `Section 1: Policy Evaluation`
   - Hoàn thành hàm `bellman_update` để cập nhật giá trị `V[s]` theo công thức Bellman cho một chính sách cố định `pi`.
   - Tích hợp hàm này vào `evaluate_policy` để thu được giá trị của chính sách hiện tại.

2. `Section 2: Policy Iteration`
   - Hoàn thành hàm `q_greedify_policy` để biến chính sách hiện tại thành chính sách greedy theo các giá trị `q` từ `V`.
   - Sử dụng `evaluate_policy` và `improve_policy` để lặp đến khi chính sách hội tụ.

3. `Section 3: Value Iteration`
   - Hoàn thành hàm `bellman_optimality_update` theo nguyên lý tối ưu của Bellman.
   - Chạy thuật toán `value_iteration` để tính giá trị tối ưu và chính sách tối ưu.

Các cell khác trong notebook dùng để kiểm tra, debug và vẽ đồ thị. Những cell này không cần chỉnh sửa trừ khi bạn muốn xem kết quả.

---

## Khái niệm chính

### 1. MDP (Markov Decision Process)
Một MDP gồm:
- `S`: tập trạng thái. Ở đây `S` là số chỗ đỗ xe đã bị chiếm.
- `A`: tập hành động. Ở đây `A` là mức giá bạn chọn.
- `P(s'|s,a)`: xác suất chuyển từ trạng thái `s` sang trạng thái `s'` khi chọn hành động `a`.
- `R(s,a,s')`: phần thưởng thu được khi chuyển từ `s` sang `s'` qua hành động `a`.
- `gamma`: hệ số chiết khấu, dùng để tính giá trị hiện tại của phần thưởng tương lai.

### 2. Value function `V`
`V[s]` cho biết giá trị kỳ vọng bắt đầu từ trạng thái `s` nếu hành động tiếp theo được chọn theo chính sách hiện tại.

### 3. Policy `pi`
`pi[s,a]` là xác suất chọn hành động `a` ở trạng thái `s`.
- Chính sách có thể là xác suất (stochastic policy).
- Trong notebook, `pi` thể hiện dưới dạng ma trận kích thước `(num_states, num_actions)`.

### 4. Bellman equation
- Công thức chính sách: `v_pi(s) = sum_a pi(a|s) sum_{s',r} p(s',r|s,a) [r + gamma * v_pi(s')]`
- Công thức tối ưu: `v_*(s) = max_a sum_{s',r} p(s',r|s,a) [r + gamma * v_*(s')]`

---

## Cấu trúc notebook và giải thích chi tiết từng phần

### Preliminaries

```python
%matplotlib inline
import numpy as np
import tools
import grader
```

- `%matplotlib inline`: dùng trong Jupyter để hiển thị biểu đồ trực tiếp trong notebook.
- `numpy`: thư viện xử lý số học.
- `tools`: mô-đun hỗ trợ môi trường `ParkingWorld` và hàm `plot`.
- `grader`: mô-đun dùng để xác thực kết quả với phần đánh giá tự động.

---

### Thiết lập môi trường ban đầu

```python
num_spaces = 3
num_prices = 3
env = tools.ParkingWorld(num_spaces, num_prices)
V = np.zeros(num_spaces + 1)
pi = np.ones((num_spaces + 1, num_prices)) / num_prices
```

- `num_spaces = 3`: có 3 chỗ đỗ xe.
- `num_prices = 3`: có 3 mức giá khác nhau.
- `env = tools.ParkingWorld(num_spaces, num_prices)`: tạo môi trường bãi đỗ xe.
- `V = np.zeros(num_spaces + 1)`: khởi tạo hàm giá trị bằng 0 cho 4 trạng thái `{0,1,2,3}`.
- `pi = np.ones((num_spaces + 1, num_prices)) / num_prices`: khởi tạo chính sách đồng đều, mọi hành động có xác suất bằng nhau.

---

### `env.transitions(state, action)`

Là phương thức quan trọng nhất của môi trường.
- Input: `state`, `action`.
- Output: mảng 2 chiều, mỗi dòng gồm `[reward, probability]` cho mỗi trạng thái kế tiếp.
- Giá trị `transitions[i,0]` là phần thưởng khi tới trạng thái `i`.
- Giá trị `transitions[i,1]` là xác suất rời trạng thái ban đầu tới trạng thái `i` khi chọn hành động đó.

Ví dụ:
```python
transitions = env.transitions(state, action)
for sp, (r, p) in enumerate(transitions):
    print(f'p(S'={sp}, R={r} | S={state}, A={action}) = {p}')
```

---

## Section 1: Policy Evaluation

### Mục đích
Tính `V` cho một chính sách cố định `pi`.

### Hàm `evaluate_policy`

```python
def evaluate_policy(env, V, pi, gamma, theta):
    delta = float('inf')
    while delta > theta:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
    return V
```

- `delta`: sai số lớn nhất giữa hai lần cập nhật liên tiếp.
- Lặp đến khi `delta <= theta` để đảm bảo giá trị đã hội tụ.
- Với mỗi trạng thái `s`, gọi `bellman_update` để cập nhật `V[s]`.

### Hàm `bellman_update`

```python
def bellman_update(env, V, pi, s, gamma):
    """Mutate ``V`` according to the Bellman update equation."""
    v_tmp=[]
    for a in env.A:
        v_tmp.append(pi[s][a]*np.sum(env.transitions(s, a)[:,1] * (env.transitions(s, a)[:,0] + (gamma * V))))
    V[s] = np.sum(v_tmp)
```

#### Giải thích từng dòng
- `v_tmp = []`: tạo danh sách tạm để lưu giá trị theo từng hành động.
- `for a in env.A:`: duyệt từng hành động khả dĩ trong trạng thái `s`.
- `pi[s][a]`: xác suất chọn hành động `a` ở trạng thái `s` theo chính sách hiện tại.
- `env.transitions(s, a)[:,1]`: lấy vectơ xác suất tới từng trạng thái kế tiếp.
- `env.transitions(s, a)[:,0]`: lấy vectơ phần thưởng tương ứng.
- `np.sum(... * (reward + gamma * V))`: tính tổng kỳ vọng phần thưởng tức thì cộng với giá trị chiết khấu của trạng thái kế tiếp.
- `pi[s][a] * ...`: nhân với xác suất chọn hành động `a`.
- `V[s] = np.sum(v_tmp)`: tổng hợp giá trị qua tất cả hành động.

#### Ý nghĩa
Hàm này chính là phiên bản in-place của cập nhật Bellman cho `v_pi(s)`.
Nó tính giá trị kỳ vọng ở trạng thái `s` khi hành động được chọn theo `pi`.

---

## Section 2: Policy Iteration

### Mục đích
Thuật toán policy iteration lặp:
1. Đánh giá chính sách hiện tại (`evaluate_policy`).
2. Cải thiện chính sách theo giá trị hiện tại (`improve_policy`).
3. Lặp lại đến khi chính sách không thay đổi.

### Hàm `improve_policy`

```python
def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in env.S:
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        if not np.array_equal(pi[s], old):
            policy_stable = False
    return pi, policy_stable
```

- `policy_stable = True`: giả định chính sách ổn định ban đầu.
- `old = pi[s].copy()`: lưu chính sách cũ của trạng thái `s`.
- `q_greedify_policy(...)`: đổi chính sách tại trạng thái `s` sang greedy.
- Nếu `pi[s]` khác `old`, ta đánh dấu chính sách chưa ổn định.
- Trả về chính sách mới và trạng thái ổn định.

### Hàm `policy_iteration`

```python
def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
    return V, pi
```

- Khởi tạo `V` và `pi` đồng đều.
- Lặp đánh giá và cải tiến đến khi chính sách hội tụ.

### Hàm `q_greedify_policy`

```python
def q_greedify_policy(env, V, pi, s, gamma):
    """Mutate ``pi`` to be greedy with respect to the q-values induced by ``V``."""
    q_tmp=[]
    for a in env.A:
        q_tmp.append(np.sum(env.transitions(s, a)[:,1] * (env.transitions(s, a)[:,0] + (gamma * V))))
    pi[s,:] = 0.
    pi[s,np.argmax(q_tmp)]=1.
```

#### Giải thích
- `q_tmp`: chứa giá trị Q(s,a) cho từng hành động.
- Với mỗi hành động `a`, tính: `E[r + gamma * V[s']]`.
- `np.argmax(q_tmp)`: chọn hành động có Q lớn nhất.
- `pi[s,:] = 0.` và `pi[s,np.argmax(q_tmp)] = 1.`: chuyển chính sách sang deterministic greedy.

#### Ý nghĩa
Hàm này biến chính sách tại trạng thái `s` thành chính sách chọn luôn hành động tốt nhất theo giá trị hiện tại.

---

## Section 3: Value Iteration

### Mục đích
Tính trực tiếp giá trị tối ưu `V*` mà không cần lặp đánh giá chính sách hoàn chỉnh tại mỗi bước.

### Hàm `value_iteration`

```python
def value_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    for s in env.S:
        q_greedify_policy(env, V, pi, s, gamma)
    return V, pi
```

- Lặp cập nhật giá trị cho đến khi sai số `delta < theta`.
- Cuối cùng xây chính sách greedy từ giá trị `V` đã hội tụ.

### Hàm `bellman_optimality_update`

```python
def bellman_optimality_update(env, V, s, gamma):
    """Mutate ``V`` according to the Bellman optimality update equation."""
    v_tmp = []
    for a in env.A:
        v_tmp.append(np.sum(env.transitions(s, a)[:,1]*(env.transitions(s, a)[:,0] + (gamma * V))))
    V[s] = np.max(v_tmp)
```

- Với mỗi hành động `a`, tính giá trị Q(s,a) giống trong policy iteration.
- `V[s] = np.max(v_tmp)`: chọn giá trị lớn nhất qua tất cả hành động.

#### Ý nghĩa
Đây là update Bellman tối ưu cho `v_*(s)`.
Nó tương đương chọn hành động tốt nhất tại mỗi trạng thái theo giả định giá trị hiện tại `V`.

---

## Ví dụ cụ thể

### 1. Policy evaluation với chính sách cố định
Cell test cho policy evaluation:

```python
num_spaces = 10
num_prices = 4
env = tools.ParkingWorld(num_spaces, num_prices)
city_policy = np.zeros((num_spaces + 1, num_prices))
city_policy[:, 1] = 1
gamma = 0.9
theta = 0.1
V = np.zeros(num_spaces + 1)
V = evaluate_policy(env, V, city_policy, gamma, theta)
```

- Chính sách `city_policy` luôn chọn hành động số 1 (giá thứ hai).
- Sau khi chạy `evaluate_policy`, ta thu được `V` kỳ vọng của chính sách đó.
- `grader.near(V, answer, 1e-2)`: kiểm tra `V` đúng đến 0.01.

### 2. Policy iteration
Cell test cho policy iteration:

```python
V, pi = policy_iteration(env, gamma, theta)
```

- Kết quả mong đợi:
  - Giá trị `V` tăng dần theo số chỗ đã chiếm, trừ những trạng thái cuối cùng có giá trị giảm vì đầy chỗ.
  - Chính sách tối ưu ở ví dụ này là chọn giá 0 khi số chỗ ít và chọn giá 3 khi sắp đầy.

### 3. Value iteration
Cell test cho value iteration:

```python
V, pi = value_iteration(env, gamma, theta)
```

- Kết quả tối ưu giống với policy iteration trong bài toán này.
- Do `Bellman optimality update` và `q_greedify_policy` cùng hướng tới giá trị/greatest action.

---

## Bài học kỹ thuật

1. `evaluate_policy` dùng Bellman cho chính sách cố định.
2. `policy_iteration` kết hợp `evaluate_policy` và cải tiến greedy để hội tụ nhanh.
3. `value_iteration` cập nhật giá trị tối ưu trực tiếp và xây chính sách sau khi hội tụ.
4. `q_greedify_policy` là cầu nối giữa giá trị `V` và chính sách `pi`.
5. `env.transitions` chứa tất cả xác suất và phần thưởng cần thiết để tính tổng kỳ vọng.

---

## Hướng dẫn đọc code theo từng dòng quan trọng

### Dòng `V = np.zeros(num_spaces + 1)`
Khởi tạo vectơ giá trị bằng 0 cho tất cả trạng thái. Với `num_spaces = 3`, ta có trạng thái `0,1,2,3`.

### Dòng `pi = np.ones((num_spaces + 1, num_prices)) / num_prices`
Khởi tạo chính sách đồng đều: mỗi trạng thái chọn mỗi hành động với xác suất bằng nhau.

### Dòng `for s in env.S:`
`env.S` là tập trạng thái. Ở môi trường này, nó là `[0, 1, ..., num_spaces]`.

### Dòng `for a in env.A:`
`env.A` là tập hành động. Ở môi trường này, nó là `[0, 1, ..., num_prices-1]`.

### Dòng `np.sum(env.transitions(s, a)[:,1] * (env.transitions(s, a)[:,0] + (gamma * V)))`
Đây là phần cốt lõi:
- `env.transitions(s,a)[:,1]`: xác suất tới mỗi trạng thái kế tiếp.
- `env.transitions(s,a)[:,0]`: phần thưởng tương ứng.
- `gamma * V`: giá trị chiết khấu của trạng thái kế tiếp.
- Nhân và cộng để ra kỳ vọng `E[r + gamma * V[s']]`.

### Dòng `pi[s,np.argmax(q_tmp)] = 1.`
Chuyển `pi[s]` thành chính sách xác định chọn hành động tốt nhất.

### Dòng `V[s] = np.max(v_tmp)`
Ở value iteration, sau khi tính giá trị cho từng hành động, chọn giá trị lớn nhất để cập nhật `V[s]`.

---

## Gợi ý nếu bạn muốn mở rộng

- Nếu muốn hiểu rõ hơn `tools.ParkingWorld`, bạn có thể tìm mã nguồn của `tools` nếu có trong thư viện cục bộ.
- Trong thực tế, `env.transitions` mô phỏng xác suất thay đổi số ô đỗ khi thay đổi giá.
- Giá trị `theta` càng nhỏ thì hội tụ chính xác hơn nhưng tốn nhiều vòng lặp hơn.

---

## Kết luận
File này đã giải thích toàn bộ ý nghĩa của bài toán, yêu cầu, các khái niệm MDP, Bellman equation, policy evaluation, policy iteration và value iteration. Tôi đã trình bày từng hàm quan trọng và giải thích chi tiết các dòng code chính trong notebook của bạn.
