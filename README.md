# Universal Data Preprocessing & Visualization (MVP++)

A **Streamlit web application** for importing, cleaning, merging, and visualizing **real-world datasets** (CSV & XLSX).  
Designed to be **universal** works across research, analytics, ML, IoT, agriculture, finance, and business datasets.

---

## 🚀 Features

### 📥 Robust Data Import
- Supports **CSV and XLSX**
- Auto-detects:
  - delimiter (`,`, `\t`, `;`, `|`)
  - header row (even after metadata blocks)
  - encoding (`utf-8`, `latin-1`)
- Safely handles malformed rows (`on_bad_lines="skip"`)
- Designed for messy exports (e.g., sensor & IoT data)

---

### 🔗 Dataset Combination
- **Single file** processing
- **Append rows** (stack multiple datasets)
- **Join/Merge datasets** on shared keys
- Optional `_source_file` column for provenance

---

### 🧹 Universal Cleaning Pipeline
- Normalize column names
- Standardize missing values (`NA`, `null`, blanks, etc.)
- Auto-infer column types:
  - numeric
  - datetime
  - categorical
- Safe ID detection (IDs never forced numeric)
- Drop or flag **high-missing columns**
- Protect critical columns from deletion
- Fully empty columns automatically removed

---

### 🧠 ML-Ready or Analysis-Ready Modes
- **ML Ready**:
  - Numeric imputation (mean / median / constant)
  - Categorical imputation (most-frequent / constant)
- **Keep Missing**:
  - Preserve NaNs for statistical analysis

---

### 🧩 Schema & Label Alignment (Optional)
- Align datasets to a user-defined schema
- Fuzzy column matching
- Optional `label / target / y` creation:
  - from existing column
  - rule-based
  - category → value mapping

---

### 📊 Smart Visualizations
Automatically **recommends charts** based on column types:

| Data Type | Visualizations |
|----------|----------------|
| Numeric | Histogram, Box plot |
| Categorical | Bar (value counts) |
| Numeric × Numeric | Scatter, Correlation |
| Datetime × Numeric | Time series |
| Categorical × Numeric | Box by category, Aggregated bar |
| Multiple numeric | Correlation heatmap |
| Any | Missingness overview |

**Extras**
- Column selection modes (Single / X-Y / Batch)
- Color palettes & grouping
- Download plots as PNG

---

### 📦 Export & Reporting
- Export cleaned data as:
  - CSV
  - XLSX
  - Parquet
- Custom missing value representation for CSV
- Download **full JSON cleaning report** (reproducibility)

---

## 🖥️ Live App
Deployable on **Streamlit Community Cloud**  
Anyone with the link can use it no login required.

---

## 🛠️ Installation (Local)

```bash
git clone https://github.com/<your-username>/universal-preprocessing-app.git
cd universal-preprocessing-app
pip install -r requirements.txt
streamlit run app.py
