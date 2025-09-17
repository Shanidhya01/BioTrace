# EDNA — Environmental DNA Analysis

Complete CSV-to-insights workflow for eDNA:

* FastAPI backend for CSV upload, parsing, prediction, and JSON/CSV exports
* React (Vite) frontend for interactive tables, charts, and reports
* Windows-first setup and scripts

Note: There are two copies of the stack. Prefer the root apps unless you intentionally work in BioTrace/.

* Primary: edna-backend and edna-frontend at repo root
* Secondary (legacy/experimental): BioTrace/edna-backend and BioTrace/edna-frontend

---

## TL;DR — Quick Start (Windows)

* Backend:

  ```
  cd d:\EDNA\edna-backend
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  ```
* Frontend:

  ```
  cd d:\EDNA\edna-frontend
  npm install
  echo VITE_API_BASE_URL=http://localhost:8000 > .env
  npm run dev
  ```
* Upload CSV in the UI (modal) or test via curl:

  ```
  curl -X POST "http://localhost:8000/upload-csv/" ^
    -H "accept: application/json" ^
    -H "Content-Type: multipart/form-data" ^
    -F "file=@D:\path\to\your\data.csv"
  ```

---

## Repository Structure

```
EDNA
├─ .gitattributes
├─ .gitignore
├─ README.md
├─ BioTrace
│  ├─ .gitignore
│  ├─ edna-backend
│  │  ├─ app.py
│  │  ├─ config.py
│  │  ├─ requirements.txt
│  │  ├─ tests_post_upload.py
│  │  ├─ tests_temp_check_root.py
│  │  ├─ train_model.py
│  │  └─ app
│  │     ├─ __init__.py
│  │     ├─ main.py
│  │     ├─ routes.py
│  │     ├─ utils.py
│  │     └─ __pycache__
│  │        ├─ __init__.cpython-311.pyc
│  │        ├─ main.cpython-311.pyc
│  │        ├─ routes.cpython-311.pyc
│  │        └─ utils.cpython-311.pyc
│  └─ edna-frontend
│     ├─ .gitignore
│     ├─ .gitignore copy
│     ├─ components.json
│     ├─ eslint.config copy.js
│     ├─ eslint.config.js
│     ├─ index.html
│     ├─ jsconfig.json
│     ├─ package-lock copy.json
│     ├─ package.json
│     ├─ README copy.md
│     ├─ README.md
│     ├─ vite.config copy.js
│     ├─ vite.config.js
│     ├─ public
│     │  └─ vite.svg
│     └─ src
│        ├─ App.css
│        ├─ App.jsx
│        ├─ index.css
│        ├─ main.jsx
│        ├─ api
│        │  └─ api.js
│        ├─ assets
│        │  └─ react.svg
│        ├─ components
│        │  ├─ Charts.jsx
│        │  ├─ DashboardPDF.jsx
│        │  ├─ DataTable.jsx
│        │  ├─ Navbar.jsx
│        │  ├─ ResultsSummary.jsx
│        │  ├─ SummaryCards.jsx
│        │  ├─ TaxonomyTable.jsx
│        │  ├─ UploadForm.jsx
│        │  └─ ui
│        │     ├─ button.jsx
│        │     ├─ card.jsx
│        │     ├─ input.jsx
│        │     └─ table.jsx
│        ├─ lib
│        │  └─ utils.js
│        └─ pages
│           ├─ Dashboard.jsx
│           └─ Home.jsx
└─ processed_data
   ├─ feature_columns.pkl
   ├─ label_encoder.pkl
   └─ scaler.pkl
```

pip install fastapi "uvicorn[standard]" pydantic python-multipart numpy pandas scikit-learn lightgbm joblib python-dotenv

```

### Environment

Optional .env overrides:

```

CORS_ORIGINS=http://localhost:5173,http://localhost:3000
TEMP_DIR=d:\EDNA\edna-backend\temp
MODEL_PATH=d:\EDNA\edna-backend\processed_data\edna_lgb_model.txt

```

Ensure upload dir exists:

```

powershell -Command "New-Item -ItemType Directory -Force -Path 'd:\EDNA\edna-backend\temp' | Out-Null"

```

### Run

```

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```

Key endpoints:

* GET /health
* POST /upload-csv/

---

## Frontend (React + Vite)

### Requirements

* Node.js 18+, npm 9+

### Install & Run

```

cd d:\EDNA\edna-frontend
npm install
echo VITE_API_BASE_URL=http://localhost:8000 > .env
npm run dev

```

Dev server: [http://localhost:5173](http://localhost:5173)

---

## CSV Format and Data Flow

Minimum CSV input:

* `sequence`: DNA sequence string
* Optional: `sample_id`, metadata

Backend response (example):

```json
{
  "status": "ok",
  "predictions": [
    {
      "sequence": "...",
      "predicted_species": "Genus species",
      "confidence_score": 0.92
    }
  ],
  "summary": {},
  "timestamp": "..."
}
```

---

## Project Workflow

### 1. Data Collection & Preprocessing

* Extracted raw eDNA sequences from **NCBI Entrez API**.
* Curated \~5,000 sequences for prototype training.
* Cleaned, standardized, and labeled sequences.

### 2. Model Training

* Trained a **LightGBM model** with preprocessing.
* Supports:

  * **Top-3 Species Prediction** with confidence scores.
  * **Novelty Clustering** to group similar unknown sequences (clusters may suggest new species).

### 3. Prediction Workflow

* **Biologist Input**: Uploads CSV with new sequences.
* **Backend (FastAPI)**:

  * Processes CSV through `/upload-csv/`.
  * Predicts species + confidence + clusters.
  * Stores/exports results as CSV/JSON.
* **Frontend (React + Vite)**:

  * Tables with sequence + top-3 predictions + confidence.
  * Taxonomy/abundance charts.
  * Export reports (CSV, charts, PDFs).

### 4. Feedback Loop

* Verified predictions & novel clusters appended to dataset.
* Model retrained periodically for improved accuracy.

---

## Typical Workflow

1. Start backend:

```
cd d:\EDNA\edna-backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

2. Start frontend:

```
cd d:\EDNA\edna-frontend
npm run dev
```

3. In the UI:

* Upload CSV → view predictions, charts, exports

---

## Troubleshooting

* **422 errors**: Request must be `multipart/form-data` with key `file`.
* **CORS issues**: Add frontend origin to CORS\_ORIGINS.
* **File not saving**: Ensure python-multipart is installed & `temp/` exists.

---

## Scripts Cheat Sheet

* Create/activate venv:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

* Install deps:

```
pip install -r requirements.txt
```

* Run backend:

```
uvicorn app.main:app --reload --port 8000
```

* Run frontend:

```
npm run dev
```

---

## Notes on BioTrace/

Contains an older copy of both backend and frontend. Use root apps unless comparing or migrating.

---

## License

Add a LICENSE file (MIT/Apache-2.0 recommended).

---

Built for a smooth CSV-to-insights eDNA workflow on Windows.
