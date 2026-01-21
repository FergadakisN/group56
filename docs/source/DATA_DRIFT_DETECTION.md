# M27: Data Drift Detection Implementation Guide

This guide documents the implementation of data drift detection for the Fish Species Classification API, enabling monitoring of input data distribution changes over time that could affect model performance.

## Overview

**M27 covers three key data drift tasks:**

1. ✅ **Check model robustness towards data drifting** - Monitor input data distribution changes
2. ✅ **Setup collection of input-output data** - Log predictions with extracted features to CSV
3. ✅ **Deploy drift detection API** - Expose `/monitoring` endpoint for drift analysis

---

## Quick Start

### 1. Start the API with Drift Logging

```bash
# Ensure dependencies are installed
uv sync --all-extras

# Start the API locally
python -m uvicorn src.group_56.api:app --host 0.0.0.0 --port 8000
```

### 2. Generate Prediction Data

```bash
# Send multiple predictions to build the database
for i in {1..20}; do
  curl -X POST http://localhost:8000/predict \
    -F file=@data/processed/test/some_fish_image.jpg
done

# Check prediction database
head -5 prediction_database.csv
```

### 3. View Drift Report

```bash
# Open monitoring endpoint in browser
open http://localhost:8000/monitoring?n_latest=50

# Or use curl
curl http://localhost:8000/monitoring?n_latest=50 > drift_report.html
open drift_report.html
```

### 4. Run Standalone Drift Analysis

```bash
# Run drift detection script
python -m src.group_56.data_drift

# View generated report
open data_drift_report.html
```

---

## Implementation Details

### Task 1: Feature Extraction from Images

Since Evidently works with structured/tabular data, we extract numerical features from images:

**Features Extracted (`extract_features.py`):**

```python
{
    "mean_brightness": 125.4,      # Average pixel intensity (0-255)
    "std_brightness": 45.2,        # Brightness variation
    "contrast": 180.0,             # Max - Min pixel values
    "sharpness": 1250.5,           # Laplacian variance (edge strength)
    "red_mean": 130.2,             # Average red channel value
    "green_mean": 120.8,           # Average green channel value
    "blue_mean": 125.1,            # Average blue channel value
    "red_std": 50.3,               # Red channel variation
    "green_std": 45.1,             # Green channel variation
    "blue_std": 40.2,              # Blue channel variation
    "aspect_ratio": 1.33,          # Width / Height
    "approx_size_kb": 192.0,       # Approximate file size
}
```

**Why These Features?**

- **Brightness/Contrast**: Detect lighting changes in images
- **Sharpness**: Identify blurry or low-quality images
- **Color Statistics**: Monitor color distribution shifts
- **Aspect Ratio**: Detect image size/shape changes

### Task 2: Prediction Logging

The API automatically logs predictions using a FastAPI background task:

**Implementation (`api.py`):**

```python
@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile, top_k: int = 5):
    # ... prediction logic ...

    # Log prediction as background task (non-blocking)
    background_tasks.add_task(log_prediction_to_csv, image, predicted_class)

    # Backup to GCS every 10 predictions
    if line_count % 10 == 0:
        background_tasks.add_task(save_to_gcs, PREDICTION_DATABASE_PATH, ...)
```

**CSV Database Format:**

```csv
timestamp,mean_brightness,std_brightness,contrast,sharpness,red_mean,green_mean,blue_mean,red_std,green_std,blue_std,aspect_ratio,approx_size_kb,prediction
2026-01-21T14:23:45.123456+00:00,125.4,45.2,180.0,1250.5,130.2,120.8,125.1,50.3,45.1,40.2,1.33,192.0,Gilt-Head Bream
2026-01-21T14:24:12.789012+00:00,115.8,40.1,165.0,1180.2,120.5,110.3,115.2,48.1,42.5,38.9,1.25,185.5,Red Mullet
```

### Task 3: Monitoring Endpoint

**Endpoint: GET `/monitoring?n_latest=100`**

**Features:**

- Analyzes latest N predictions (default: 100)
- Generates Evidently HTML report with:
  - **Data Drift Preset**: Feature distribution changes
  - **Data Quality Preset**: Missing values, duplicates, outliers
  - **Target Drift Preset**: Prediction distribution changes
- Returns interactive HTML dashboard

**Example Response:**

```html
<!-- Evidently Report with:
  - Dataset Summary
  - Data Drift Detection (per feature)
  - Data Quality Metrics
  - Target Drift Analysis
  - Feature Distributions (histograms)
-->
```

**Workflow:**

1. Load prediction database CSV
2. Split data: first 50% as reference, last 50% as current
3. Run Evidently drift detection
4. Return HTML report

---

## Drift Detection Concepts

### What is Data Drift?

Data drift occurs when the statistical properties of input data change over time, causing model performance degradation.

**Types of Drift:**

1. **Covariate Shift**: Input distribution changes (P(X) changes, P(Y|X) stays same)
   - Example: Brighter images in summer vs winter

2. **Prior Probability Shift**: Label distribution changes (P(Y) changes)
   - Example: More requests for certain fish species in fishing season

3. **Concept Drift**: Relationship between X and Y changes (P(Y|X) changes)
   - Example: New camera type captures different image characteristics

### Detection Methods

**Statistical Tests (Evidently uses):**

- **Kolmogorov-Smirnov test**: For continuous features
- **Chi-squared test**: For categorical features
- **Population Stability Index (PSI)**: Overall drift score
- **Wasserstein distance**: Distribution similarity

**Thresholds:**

- `drift_score > 0.5`: Significant drift detected
- `drift_share > 0.3`: More than 30% of features drifted
- `psi > 0.2`: High drift, `psi > 0.1`: Moderate drift

---

## Testing Data Drift Detection

### Test 1: Verify Feature Extraction

```bash
python << EOF
from PIL import Image
from src.group_56.extract_features import extract_image_features

# Load test image
image = Image.open("data/processed/test/some_fish_image.jpg")

# Extract features
features = extract_image_features(image)

# Print
for key, value in features.items():
    print(f"{key}: {value:.2f}")
EOF
```

**Expected Output:**

```txt
mean_brightness: 125.40
std_brightness: 45.20
contrast: 180.00
sharpness: 1250.50
...
```

### Test 2: Simulate Data Drift

```python
from PIL import Image, ImageEnhance
import os

# Send normal images
for i in range(20):
    image_path = f"data/processed/test/image_{i}.jpg"
    os.system(f"curl -X POST http://localhost:8000/predict -F file=@{image_path}")

# Send altered images (simulate drift)
for i in range(20):
    image = Image.open(f"data/processed/test/image_{i}.jpg")

    # Increase brightness by 50%
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(1.5)

    bright_image.save("/tmp/bright_image.jpg")
    os.system("curl -X POST http://localhost:8000/predict -F file=@/tmp/bright_image.jpg")
```

**Check Drift Report:**

```bash
open http://localhost:8000/monitoring?n_latest=40
```

**Expected:** Report should show drift in `mean_brightness`, `red_mean`, `green_mean`, `blue_mean`.

### Test 3: Target Drift Detection

```bash
# Send many images of same species to simulate target drift
for i in {1..30}; do
  curl -X POST http://localhost:8000/predict \
    -F file=@data/processed/test/gilt_head_bream_*.jpg
done

# View monitoring
open http://localhost:8000/monitoring?n_latest=50
```

**Expected:** Target drift detected in prediction distribution.

---

## Interpreting Drift Reports

### Dashboard Sections

#### 1. Dataset Summary

```txt
Reference Dataset: 50 rows × 13 columns
Current Dataset: 50 rows × 13 columns
Column Types: 12 numerical, 1 categorical
```

#### 2. Data Drift

| Feature | Drift Score | Status | Test |
| --------- | ------------- | -------- | ------ |
| mean_brightness | 0.12 | ✅ OK | K-S test |
| contrast | 0.45 | ⚠️ Warning | K-S test |
| sharpness | 0.78 | ❌ Drift | K-S test |

**Interpretation:**

- **✅ OK**: Feature distribution is stable
- **⚠️ Warning**: Minor drift, monitor closely
- **❌ Drift**: Significant drift, consider retraining

#### 3. Data Quality

```txt
Missing Values: 0
Duplicates: 2
Constant Features: 0
Almost Constant Features: 0
```

#### 4. Target Drift

```txt
Prediction Distribution Drift: 0.35 (Moderate)
Most Common Class (Reference): Gilt-Head Bream (25%)
Most Common Class (Current): Red Mullet (40%)
```

**Interpretation:** Model is predicting Red Mullet more frequently in current data.

---

## Monitoring Best Practices

### 1. Set Drift Thresholds

```python
# In data_drift.py, customize tests
from evidently.tests import TestShareOfDriftedColumns

test_suite = TestSuite(tests=[
    TestShareOfDriftedColumns(lt=0.3),  # Alert if >30% features drift
])
```

### 2. Regular Monitoring Schedule

**Recommended Frequency:**

- High traffic: Hourly or daily
- Medium traffic: Weekly
- Low traffic: Monthly

**Automate with cron:**

```bash
# Check drift daily at 2 AM
0 2 * * * python -m src.group_56.data_drift && mail -s "Drift Report" you@example.com < data_drift_report.html
```

### 3. Alert on Critical Drift

```python
# In data_drift.py
results = test_suite.as_dict()
drift_share = results["summary"]["all_passed"]

if not drift_share:
    send_alert("Critical drift detected!")
    trigger_retraining()
```

### 4. Compare to Training Data

For production, extract features from training data:

```python
from src.group_56.data import FishDataset
from src.group_56.extract_features import extract_image_features
import pandas as pd

# Load training images
train_dataset = FishDataset("data/processed/train")

# Extract features
train_features = []
for img_path, label in train_dataset.samples:
    image = Image.open(img_path)
    features = extract_image_features(image)
    features["prediction"] = label
    train_features.append(features)

# Save as reference
ref_df = pd.DataFrame(train_features)
ref_df.to_csv("reference_data.csv", index=False)
```

Then use in monitoring:

```python
reference_data = pd.read_csv("reference_data.csv")
current_data = load_current_data("prediction_database.csv", n_latest=100)
report = generate_drift_report(reference_data, current_data)
```

---

## Deployment to Cloud Run

### Update Dockerfile

Ensure `evidently` and `scikit-learn` are in requirements.txt (already done).

### Deploy with Monitoring

```bash
# Build and push
docker build -f dockerfiles/api.dockerfile -t us-central1-docker.pkg.dev/fish-mlops/fish/fish-classifier:v5 .
docker push us-central1-docker.pkg.dev/fish-mlops/fish/fish-classifier:v5

# Deploy
gcloud run deploy fish-classifier-api \
  --image us-central1-docker.pkg.dev/fish-mlops/fish/fish-classifier:v5 \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated \
  --update-env-vars MODEL_PATH=gs://fish-mlops-bucket/model.pt,GCS_BUCKET=fish-mlops \
  --project fish-mlops
```

### Access Monitoring in Production

```bash
SERVICE_URL=$(gcloud run services describe fish-classifier-api \
  --region us-central1 \
  --format='value(status.url)' \
  --project fish-mlops)

# View drift report
open "$SERVICE_URL/monitoring?n_latest=200"
```

---

## Limitations & Future Work

### Current Limitations

1. **Reference Data**: Currently uses first 50% of production data as reference. Should use actual training data features.

2. **Feature Extraction**: Limited to basic image statistics. Could add:
   - Deep learning embeddings (ResNet features before final layer)
   - CLIP embeddings for semantic features
   - EXIF metadata (camera, timestamp, location)

3. **Marginal Distributions Only**: Evidently analyzes features independently. Multivariate drift (joint distributions) not detected.

### Improvements

**1. Add Deep Feature Extraction:**

```python
def extract_deep_features(image, model):
    """Extract features from penultimate layer of model."""
    with torch.no_grad():
        features = model.avgpool(model.layer4(image))
    return features.flatten().cpu().numpy()
```

**2. Implement Multivariate Drift Detection:**

```python
from scipy.stats import chi2_contingency
from sklearn.metrics import maximum_mean_discrepancy

# MMD test for multivariate drift
mmd_score = maximum_mean_discrepancy(reference_features, current_features)
```

**3. Scheduled Cloud Function:**

```python
# deploy as Cloud Function that runs daily
def check_drift(request):
    current_data = download_from_gcs("gs://fish-mlops/monitoring/prediction_database.csv")
    reference_data = download_from_gcs("gs://fish-mlops/reference_data.csv")

    report = generate_drift_report(reference_data, current_data)

    if drift_detected(report):
        send_email_alert()
        trigger_retraining_pipeline()
```

---

## Troubleshooting

### Issue: "No predictions logged"

**Solution:**

```bash
# Check if CSV exists
ls -lh prediction_database.csv

# If not, make some predictions
curl -X POST http://localhost:8000/predict -F file=@test_image.jpg
```

### Issue: "Insufficient data for drift detection"

**Solution:** Need at least 10 predictions. Generate more:

```bash
for i in {1..15}; do
  curl -X POST http://localhost:8000/predict -F file=@data/processed/test/*.jpg
done
```

### Issue: Drift report shows all features drifted

**Cause:** Using production data as its own reference (first half vs second half with small sample size).

**Solution:** Extract features from training data and use as reference:

```bash
python << EOF
from pathlib import Path
from PIL import Image
import pandas as pd
from src.group_56.extract_features import extract_image_features

train_dir = Path("data/processed/train")
features_list = []

for img_path in list(train_dir.rglob("*.jpg"))[:100]:
    image = Image.open(img_path)
    features = extract_image_features(image)
    features["prediction"] = img_path.parent.name
    features_list.append(features)

df = pd.DataFrame(features_list)
df.to_csv("reference_data.csv", index=False)
print(f"Extracted {len(df)} reference samples")
EOF
```

### Issue: Monitoring endpoint times out

**Cause:** Too many predictions to analyze.

**Solution:** Use `n_latest` parameter:

```bash
curl "http://localhost:8000/monitoring?n_latest=50"
```

---

## M27 Completion Checklist

- [x] Extract structured features from images
  - [x] Brightness, contrast, sharpness
  - [x] Color channel statistics
  - [x] Aspect ratio and size

- [x] Implement prediction logging
  - [x] Background task for non-blocking logging
  - [x] CSV database with timestamps
  - [x] GCS backup every 10 predictions

- [x] Setup drift detection with Evidently
  - [x] Data drift preset (feature distributions)
  - [x] Data quality preset (missing values, outliers)
  - [x] Target drift preset (prediction distribution)

- [x] Deploy monitoring endpoint
  - [x] `/monitoring` GET endpoint
  - [x] Configurable sample size (`n_latest`)
  - [x] Interactive HTML reports

- [x] Testing & Documentation
  - [x] Local testing procedures
  - [x] Drift simulation examples
  - [x] Interpretation guide

---

## References

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [DTU MLOps - Data Drifting](https://skaftenicki.github.io/dtu_mlops/s8_monitoring/data_drifting/)
- [Understanding Data Drift](https://www.evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift)
- [NannyML](https://github.com/NannyML/nannyml) - Alternative drift detection framework
- [WhyLogs](https://github.com/whylabs/whylogs) - Data logging for ML
