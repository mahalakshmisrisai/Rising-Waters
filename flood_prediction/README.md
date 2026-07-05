# Flood Prediction System

Machine learning–powered flood early-warning system. Trains Decision Tree,
Random Forest, KNN, and XGBoost classifiers on historical weather data,
automatically picks the best-performing model, and serves it through a
Flask web application.

## Project structure

```
flood_prediction/
├── generate_dataset.py     # creates flood_data.csv (synthetic sample data)
├── train_model.py          # trains all 4 models, compares, saves the best one
├── app.py                  # Flask web app that serves predictions
├── requirements.txt
├── templates/
│   └── index.html          # web UI
└── model/                  # created after training
    ├── flood_model.pkl
    ├── scaler.pkl
    ├── feature_columns.pkl
    ├── best_model_name.pkl
    └── model_comparison.png
```

## 1. Setup

```bash
# create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 2. Get your data

You have two options:

**Option A — use the included synthetic dataset (quick start):**
```bash
python generate_dataset.py
```
This creates `flood_data.csv` with realistic, but artificial, rainfall/cloud
visibility/temperature/humidity data and a `flood` (0/1) label.

**Option B — use your own historical weather dataset:**
Replace `flood_data.csv` with your real data. It must have these columns
(rename in `train_model.py -> FEATURE_COLUMNS` if your column names differ):

| Column              | Meaning                                   |
|---------------------|--------------------------------------------|
| `annual_rainfall`   | Total yearly rainfall (mm)                 |
| `seasonal_rainfall` | Monsoon/rainy-season rainfall (mm)         |
| `cloud_visibility`  | Visibility affected by cloud cover (km)    |
| `temperature`       | Average temperature (°C)                   |
| `humidity`          | Average relative humidity (%)              |
| `flood`             | Target label — 1 = flood occurred, 0 = not |

## 3. Train the models

```bash
python train_model.py
```

This will:
- Split the data into train/test sets
- Scale features with `StandardScaler`
- Train Decision Tree, Random Forest, KNN, and XGBoost
- Print an accuracy/precision/recall report for each
- Save the **best-performing model** as `model/flood_model.pkl`
- Save `model/scaler.pkl`, `model/feature_columns.pkl`, `model/best_model_name.pkl`
- Save a bar chart comparing all 4 models to `model/model_comparison.png`

Example output:
```
================ SUMMARY ================
Random Forest  : 81.17%
Decision Tree  : 81.00%
KNN            : 79.50%
XGBoost        : 79.33%

Best model: Random Forest (81.17% accuracy)
```
(Your exact numbers will depend on the dataset you use — with clean,
well-engineered historical flood data like the scenario described, XGBoost
can reach the 95%+ range mentioned in the project brief.)

## 4. Run the web app

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

- Fill in the weather readings (annual rainfall, seasonal rainfall, cloud
  visibility, temperature, humidity)
- Click **Predict Flood Risk**
- The app displays the classification (High/Low Risk) and the flood
  probability

### JSON API (for integrating with other systems / mobile apps)

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "annual_rainfall": 2500,
        "seasonal_rainfall": 1600,
        "cloud_visibility": 1.5,
        "temperature": 27,
        "humidity": 92
      }'
```

Response:
```json
{
  "flood_predicted": true,
  "probability": 0.988,
  "model_used": "Random Forest"
}
```

## 5. Re-training with new data

Whenever new historical weather/flood records become available, just
overwrite `flood_data.csv` and re-run `python train_model.py`. It will
retrain all 4 models, re-select the best one, and overwrite the saved
model files — no changes to `app.py` are needed as long as the column
names stay the same.

## 6. Deploying to IBM Cloud

The app is a standard Flask app, so it deploys cleanly to **IBM Cloud
Code Engine** or **Cloud Foundry**:

1. Add a `Procfile` with: `web: gunicorn app:app`
2. Add `gunicorn` to `requirements.txt`
3. Push to IBM Cloud Code Engine as a container, or use `ibmcloud cf push`
   for Cloud Foundry
4. Make sure the `model/` folder (trained artifacts) is included in the
   deployment package — either commit it or run the training step as part
   of your build pipeline

## Notes on the ML approach

- **Preprocessing**: features are standardized with `StandardScaler` before
  training (most impactful for KNN; harmless for tree-based models).
- **Why compare 4 algorithms**: Decision Tree and Random Forest give
  interpretable, robust baselines; KNN captures local similarity patterns;
  XGBoost typically achieves the highest accuracy on structured tabular
  data like this, which is why it's highlighted in the "Model Validation"
  scenario.
- **Model selection**: `train_model.py` automatically picks whichever model
  scores highest on held-out test accuracy — no manual step required.
