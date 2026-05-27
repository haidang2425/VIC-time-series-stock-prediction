# Phân Tích & Dự Báo Giá Cổ Phiếu VIC (Vingroup) bằng Mô Hình Chuỗi Thời Gian

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Prophet](https://img.shields.io/badge/Meta-Prophet-0668E1?style=for-the-badge&logo=meta&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-HPO-6DB3F2?style=for-the-badge)
![ARIMA](https://img.shields.io/badge/Statsmodels-ARIMA-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

**Ứng dụng Time Series Analysis & Machine Learning để dự báo biến động giá cổ phiếu VIC.VN trên thị trường chứng khoán Việt Nam**

</div>

---

|  |  |
|---|---|
| **Môn học** | AIE301m – Ứng dụng Học máy trong Tài chính |
| **Sinh viên** | Trịnh Hải Đăng – HE194363 |
| **Trường** | Đại học FPT |
| **Thời gian** | Học kỳ 8, Năm học 2025–2026 |

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Phương pháp luận](#2-phương-pháp-luận)
3. [Kiến trúc & Pipeline](#3-kiến-trúc--pipeline)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Hướng dẫn cài đặt](#5-hướng-dẫn-cài-đặt)
6. [Tiêu chí đánh giá](#6-tiêu-chí-đánh-giá)
7. [Kết quả thực nghiệm](#7-kết-quả-thực-nghiệm)
8. [Thảo luận & Hướng phát triển](#8-thảo-luận--hướng-phát-triển)
9. [Tuyên bố miễn trách](#9-tuyên-bố-miễn-trách)

---

## 1. Tổng quan dự án

Dự án xây dựng một **pipeline dự báo chuỗi thời gian (Time Series Forecasting Pipeline)** hoàn chỉnh nhằm mô hình hóa và dự báo giá đóng cửa của mã cổ phiếu **VIC.VN** (Tập đoàn Vingroup) — một trong những bluechip hàng đầu trên sàn HOSE.

Dữ liệu giá lịch sử được thu thập trực tiếp từ **Yahoo Finance API** và được kiểm định trên giai đoạn thực tế từ `15/05/2025` đến `26/05/2026`.

**Điểm nổi bật của dự án:**

- Áp dụng chiến lược **Walk-Forward Backtesting** nghiêm ngặt — mô hình **không bao giờ được tiếp xúc với dữ liệu tương lai** trong quá trình huấn luyện.
- So sánh có hệ thống **4 mô hình** từ cổ điển thống kê đến hiện đại, với phân tích định lượng rõ ràng.
- Tích hợp **Optuna** — framework tối ưu hóa siêu tham số (Hyperparameter Optimization) theo hướng out-of-sample, giải quyết điểm yếu cốt lõi của Auto-ARIMA truyền thống.
- Kết hợp đầu ra của nhiều mô hình qua **Ensemble Learning** để tối thiểu hóa phương sai sai số tổng thể.

---

## 2. Phương pháp luận

Dự án được phát triển tuần tự qua **4 cấp độ mô hình**, mỗi cấp khắc phục điểm yếu của cấp trước, tạo thành một lộ trình nghiên cứu có tư duy:

### Cấp độ 1 — Auto-ARIMA (Baseline)

> *"Bắt đầu từ nền tảng cổ điển để thiết lập đường cơ sở (baseline) so sánh."*

- Sử dụng thuật toán **Grid Search** tự động tìm kiếm tổ hợp siêu tham số $(p, d, q)$ tối ưu dựa trên tiêu chí **AIC** (Akaike Information Criterion).
- **Hạn chế được ghi nhận:** Tối ưu hóa trên tập huấn luyện dễ dẫn đến **Overfitting** hoặc mô hình hội tụ về **Random Walk** khi dữ liệu có nhiễu cao.

### Cấp độ 2 — Optuna-ARIMA (Out-of-sample Optimization)

> *"Tối ưu hóa mô hình dựa trên khả năng dự báo thực tế, không phải trên dữ liệu đã biết."*

- Tích hợp framework **Optuna** (Bayesian / TPE Sampler) để thay thế Grid Search truyền thống.
- Tách riêng **10% dữ liệu** làm `Validation Set` ẩn. Optuna tìm bộ $(p, d, q)$ tối thiểu hóa **RMSE trên Validation Set** — tức là tìm mô hình *tốt nhất cho tương lai*, không phải tốt nhất cho quá khứ.

### Cấp độ 3 — Meta Prophet (Decomposable Time Series Model)

> *"Bắt các xu hướng phi tuyến tính mà ARIMA bỏ sót."*

- Mô hình chuỗi thời gian phân rã (**Decomposable Time Series**) do nhóm **Core Data Science của Meta** phát triển.
- Mạnh mẽ trong việc phát hiện **Trend Changepoints** — các điểm gãy xu hướng đột ngột đặc trưng của thị trường tài chính.
- Không yêu cầu dữ liệu phải **stationary**, phù hợp với chuỗi giá tuyệt đối (absolute price).

### Cấp độ 4 — Ensemble Model (State-of-the-art)

> *"Kết hợp thế mạnh của nhiều mô hình để vượt qua giới hạn của từng mô hình đơn lẻ."*

- Áp dụng triết lý **Ensemble Learning**: lấy **trung bình trọng số** dự báo của (Auto-ARIMA + Prophet).
- **Mục tiêu:** ARIMA đóng góp khả năng bắt **cấu trúc tự hồi quy (Lag structure)**; Prophet đóng góp khả năng bám sát **xu hướng dài hạn (Trend)**. Kết hợp hai nguồn thông tin bổ sung nhau giúp giảm phương sai sai số tổng thể.

---

## 3. Kiến trúc & Pipeline

```
Yahoo Finance API
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                   DATA INGESTION                        │
│  Thu thập dữ liệu lịch sử VIC.VN → Xử lý missing       │
│  values → Kiểm định tính dừng (ADF Test)                │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  DATA SPLITTING                         │
│                                                         │
│   Train Set (80%)  │  Validation Set (10%)  │  Test (10%)│
│   (Huấn luyện)     │  (Optuna Tuning)       │  (Ẩn)      │
└─────────────────────────┬───────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      Auto-ARIMA    Optuna-ARIMA    Meta Prophet
            │             │             │
            └─────────────┼─────────────┘
                          ▼
                   Ensemble Model
                  (Weighted Average)
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              EVALUATION ON TEST SET                     │
│         RMSE  │  MAE  │  MAPE (Target: < 5%)           │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Cấu trúc thư mục

```
VIC-time-series-stock-prediction/
│
├── notebooks/
│   └── VIC_Stock_ARIMA_Analysis.ipynb   # Pipeline chính: EDA → Modeling → Evaluation
│
├── requirements.txt                      # Dependencies (pip freeze)
├── .gitignore                            # Loại trừ .venv, cache, checkpoints
└── README.md                             # Tài liệu kỹ thuật (file này)
```

---

## 5. Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Python **3.9+**
- pip **22+**
- (Khuyến nghị) Hệ điều hành: macOS / Linux / Windows 10+

### Bước 1 — Clone repository

```bash
git clone https://github.com/haidang2425/VIC-time-series-stock-prediction.git
cd VIC-time-series-stock-prediction
```

### Bước 2 — Tạo và kích hoạt môi trường ảo

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Bước 3 — Cài đặt dependencies

> **Lưu ý:** `prophet` và `optuna` có các thành phần biên dịch C++. Đảm bảo hệ thống có **build tools** phù hợp (Visual Studio Build Tools trên Windows, `gcc` trên Linux/macOS).

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Bước 4 — Khởi chạy Jupyter Notebook

```bash
jupyter notebook
```

Mở file `notebooks/VIC_Stock_ARIMA_Analysis.ipynb` và chạy tuần tự từng cell (`Run All`).

---

## 6. Tiêu chí đánh giá

Tất cả các mô hình được đánh giá **chỉ trên Test Set** — tập dữ liệu hoàn toàn bị ẩn trong suốt quá trình huấn luyện và tinh chỉnh.

| Chỉ số | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Phạt nặng các sai lệch lớn; nhạy cảm với outliers |
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Sai lệch tuyệt đối trung bình (đơn vị: VNĐ) |
| **MAPE** | $\frac{100\%}{n}\sum\left\|\frac{y_i - \hat{y}_i}{y_i}\right\|$ | Sai số theo phần trăm; **mục tiêu: MAPE < 5%** |

---

## 7. Kết quả thực nghiệm

> *Bảng kết quả sẽ được cập nhật sau khi hoàn thành vòng Backtesting trên tập Test Set thực tế.*

| Mô hình | RMSE (VNĐ) | MAE (VNĐ) | MAPE (%) | Ghi chú |
|---------|:---------:|:--------:|:-------:|---------|
| Auto-ARIMA | 11,007 | 9,316 | 4.28% | Baseline |
| Optuna-ARIMA (0,2,0) | 2,321 | 2,086 | **0.95%** | Best single model |
| Meta Prophet | 7,330 | 5,888 | 2.71% | |
| **Ensemble** (ARIMA + Prophet) | **9,038** | **6,925** | **3.20%** | Combined |

> **Nhận xét:** Optuna-ARIMA đạt MAPE **0.95%** — vượt xa mục tiêu 5% đề ra, đồng thời là mô hình đơn lẻ tốt nhất trong pipeline. Kết quả cho thấy chiến lược tối ưu hóa out-of-sample bằng Optuna hiệu quả rõ rệt so với Auto-ARIMA truyền thống (MAPE giảm từ 4.28% xuống 0.95%, tương đương **77.8%**). Tập Test bao gồm 7 ngày giao dịch thực tế từ 16/05/2026 đến 26/05/2026.

---

## 8. Thảo luận & Hướng phát triển

### Giới hạn của mô hình & Lý thuyết Thị trường Hiệu quả (EMH)

Mọi mô hình kỹ thuật (ARIMA, Prophet) đều dựa trên giả định **lịch sử có xu hướng lặp lại**. Tuy nhiên, theo **Efficient Market Hypothesis (EMH)**, giá cổ phiếu phản ánh tức thời toàn bộ thông tin công khai — điều này về mặt lý thuyết khiến việc dự báo vượt trội thị trường là bất khả thi. Trên thực tế, thị trường Việt Nam còn chịu tác động mạnh từ các **biến ngoại sinh (Exogenous Variables)** như chính sách lãi suất, báo cáo tài chính doanh nghiệp, và tâm lý nhà đầu tư — những yếu tố không được mã hóa trong chuỗi giá lịch sử.

### Hướng nâng cấp đề xuất

**1. Mô hình hóa trên Log Returns thay vì giá tuyệt đối**

Về mặt Toán tài chính (Quantitative Finance), chuỗi giá tuyệt đối thường **không dừng (non-stationary)**. Phiên bản tiếp theo sẽ mô hình hóa trên **chuỗi lợi suất logarit**:

$$R_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Chuỗi $R_t$ mang tính **dừng (stationary)** tự nhiên, giúp các mô hình tự hồi quy hoạt động ổn định và chính xác hơn về mặt toán học.

**2. Mô hình hóa biến động (Volatility Modeling — GARCH)**

Tích hợp họ mô hình **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** để dự báo **biên độ rủi ro (phương sai có điều kiện)** thay vì chỉ dự báo giá trị kỳ vọng. Đây là bước thiết yếu trong định giá quyền chọn và quản lý rủi ro danh mục.

**3. Tích hợp biến ngoại sinh (ARIMAX / Prophet with Regressors)**

Bổ sung các regressors như: chỉ số VN-Index, tỷ giá USD/VND, lãi suất liên ngân hàng (VNIBOR), và khối lượng giao dịch — nhằm cải thiện khả năng dự báo của mô hình trong các giai đoạn thị trường biến động mạnh.

**4. Triển khai hệ thống (MLOps)**

Đóng gói mô hình dưới dạng **REST API** với `FastAPI` và xây dựng **Dashboard trực quan hóa** real-time bằng `Streamlit`, tạo thành một sản phẩm end-to-end có thể deploy lên cloud (Railway / Render / AWS).

---

## 9. Tuyên bố miễn trách

> Dự án này được phát triển **hoàn toàn phục vụ mục đích nghiên cứu và học thuật** trong khuôn khổ môn học. Các kết quả dự báo **không phải là lời khuyên hay khuyến nghị đầu tư tài chính**. Người đọc không nên đưa ra quyết định đầu tư dựa trên bất kỳ nội dung nào trong dự án này.

---

<div align="center">

Made with dedication by **Trịnh Hải Đăng** (HE194363)

*"Predicting the market is hard. Understanding why it's hard is the real insight."*
pip freeze > requirements.txt
</div>
