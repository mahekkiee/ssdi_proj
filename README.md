# 📈 NSE Stock Sector Analysis — Streamlit App

Interactive dashboard presenting the statistical analysis of 9 NSE stocks across
IT, Banking and Pharma sectors (Linear Regression, ANOVA, Tukey HSD, MANOVA).

---

## 🚀 Deploy in 5 minutes (free)

### Step 1 — Export your data from the notebook

Open your Jupyter notebook and, **after the cell that builds `df`** (around cell 7,
after the `Daily_Return` / `Daily_Range_Pct` columns are added), add and run this line:

```python
df.to_csv('data.csv', index=False)
```

This creates a single `data.csv` file. Copy it into this project folder.

### Step 2 — Put these files in a GitHub repo

You should have these 4 files in one folder:

```
your-repo/
├── app.py
├── requirements.txt
├── data.csv          ← the one you just exported
└── README.md
```

Create a new GitHub repo, then push the files:

```bash
git init
git add .
git commit -m "Initial Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> Don't have `git` set up? Just drag-and-drop these 4 files into a new repo on
> github.com — the web interface works fine.

### Step 3 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub
2. Click **"New app"**
3. Pick your repo, branch = `main`, main file = `app.py`
4. Click **Deploy**

Your app will be live at `https://<your-app-name>.streamlit.app` in about a minute.

---

## 💻 Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📂 What's in the app

| Page | Contents |
|---|---|
| **Overview** | Dataset summary, sample rows, rows per ticker |
| **Exploratory Analysis** | Summary stats, correlation heatmap, price-over-time, sector boxplots |
| **Linear Regression** | Model comparison table, VIF diagnostics, refined model, residual plots |
| **One-way ANOVA** | Returns & volatility by sector + Tukey HSD visualizations |
| **Two-way ANOVA** | Sector × Day-of-week interaction |
| **MANOVA** | Joint test on OHLC + per-variable Tukey follow-up |
| **Conclusions** | Summary table, use cases, limitations |

---

*Built for the SSDI module · Data from NSE India*
