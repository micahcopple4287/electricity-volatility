# US Electricity Price Volatility Explorer

An end-to-end data science project analyzing retail electricity prices across five US states using data from the EIA API. Includes a full data pipeline, exploratory analysis, machine learning models, and an interactive Streamlit dashboard.

## Research Question
**What factors predict electricity price volatility across US states, and can we forecast when prices will spike?**

## Key Findings

- **Generation mix is the strongest structural predictor** of price level. States with renewable/hydro-heavy grids (CA, NY) have significantly higher average prices than fossil-fuel-heavy states (TX, CO).
- **Lag features and rolling statistics dominate short-term prediction.** The 3-month rolling mean alone explains the majority of month-to-month price variation.
- **The model performs well on gradual-trend states but breaks down on regime changes.** TX and CO achieve under 1% prediction error. CA and NY experienced a structural post-2021 price surge that historical patterns cannot anticipate — an important honest limitation of the approach.
- **Spike detection achieves 79% accuracy** using a rolling-window spike definition, with 76% precision on true spike months.

## Stack

| Layer | Tools |
|---|---|
| Data acquisition | Python, Requests, EIA API v2 |
| Data processing | Pandas, NumPy |
| Modeling | XGBoost, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Environment | Miniconda, Python 3.11 |

## Project Structure

```
electricity-volatility/
├── app/
│   └── dashboard.py        # Streamlit dashboard
├── data/
│   ├── raw/                # EIA API outputs (CSV)
│   └── processed/          # Engineered features, trained models
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory analysis
│   └── 02_model.ipynb      # Feature engineering + modeling
├── src/
│   └── data_fetcher.py     # EIA API scraper
├── .env                    # API key (not committed)
├── .gitignore
└── README.md
```

## Models

### Regression: Predict exact price (cents/kWh)
- **Algorithm:** XGBoost Regressor
- **Features:** Lag prices (1, 2, 3, 12 months), rolling mean/std (3m, 12m), cyclical month encoding, YoY change, state encoding
- **Train/test split:** Chronological 80/20 (trained on 2001–2021, tested on 2021–2026)
- **MAE:** 2.03 cents/kWh on an average test price of 19.59 cents/kWh (~10% error)
- **RMSE:** 3.91 cents/kWh (higher than MAE, indicating spike underprediction)

### Classifier: Predict price spikes (top 20% relative to trailing 12 months)
- **Algorithm:** XGBoost Classifier
- **Spike definition:** Rolling 80th percentile of trailing 12 months (avoids data leakage from global percentile)
- **Accuracy:** 79%
- **Spike precision:** 76% | **Spike recall:** 61%

## Running Locally

1. Clone the repo
```bash
git clone https://github.com/micahcopple4287/electricity-volatility.git
cd electricity-volatility
```

2. Create and activate the environment
```bash
conda create -n electricity-analysis python=3.11 pandas requests numpy jupyter -y
conda activate electricity-analysis
pip install python-dotenv scikit-learn xgboost streamlit matplotlib seaborn joblib
```

3. Add your EIA API key to `.env`
```
EIA_API_KEY=your_key_here
```
Get a free key at [api.eia.gov](https://api.eia.gov)

4. Pull the data
```bash
python src/data_fetcher.py
```

5. Run the notebooks in order
- `notebooks/01_eda.ipynb`
- `notebooks/02_model.ipynb`

6. Launch the dashboard
```bash
streamlit run app/dashboard.py
```

## Limitations & Honest Notes

- **Only 5 states:** Expanding to all 50 would require more API calls and richer feature engineering
- **No weather data:** Temperature is a major demand driver not included in this version
- **Regime changes break lag models:** The post-2021 energy price surge in CA and NY is not predictable from historical price patterns alone; external signals (policy changes, grid investment data) would be needed
- **Generation mix is cross-sectional:** The correlation between renewables and high prices reflects structural state differences, not a causal relationship

## Contact

Micah Copple | [micahcopple.netlify.app](https://micahcopple.netlify.app) | micahcopple@gmail.com
