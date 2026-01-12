# MLOps Project Plan: Fish Species Classification

## 📌 Project Overview

**Objective:** Develop a production-grade MLOps pipeline for fine-grained classification of 468 fish species.
**Key Constraints:**

* **Storage:** Dataset must remain < 1GB (Utilization of `cropped` images only).
* **Data Imbalance:** Handling rare classes (3-4 images/species) vs. common classes.
* **Robustness:** Addressing domain shift between "Controlled" and "In-Situ" environments.

---

## 📅 Phase 1: Foundation & Data Management

**Goal:** Establish a reproducible, version-controlled data pipeline and robust project structure.

### 1.1 Project Initialization & Structure

* **Tools:** `Cookiecutter`, `Git`, `Virtual Environment`.
* **Tasks:**
  * Initialize repository using the DTU MLOps template:
        `cookiecutter https://github.com/SkafteNicki/mlops-template`
  * Create a clean virtual environment and `requirements.txt` (or `pyproject.toml`).
  * **Data Organization:**
    * Discard `raw` (background noise) and `numbered` (redundant) folders.
    * Place `cropped` images and `final_all_index.txt` into `data/raw/`.

### 1.2 Data Version Control (DVC)

* **Tools:** `DVC`, `Google Cloud Storage (GCS)`.
* **Tasks:**
  * Initialize DVC: `dvc init`.
  * Track the dataset:
    * `dvc add data/raw/cropped`
    * `dvc add data/raw/final_all_index.txt`
  * Configure remote storage (GCP Bucket) to share data across the team without bloating the git repo.

### 1.3 Data Ingestion & Preprocessing

* **Tools:** `Pandas`, `PyTorch`.
* **Script:** `src/data/make_dataset.py`
* **Tasks:**
  * **Parsing:** Load `final_all_index.txt` to map filenames to metadata.
  * **Cleaning:** Filter out rows where condition is `rubbish`, `sketches`, or `uncontrolled`.
  * **Stratified Splitting (Crucial):**
    * Group data by `species_id`.
    * **Rare Classes (<3 images):** Force 100% to Training set (cannot test on unseen classes).
    * **Common Classes (>=3 images):** Ensure at least 1 example in Train, Val, and Test.
  * **Output:** Save processed tensors or file lists (`train.pt`, `test.pt`) to `data/processed/`.

### 1.4 Code Quality Assurance

* **Tools:** `Pre-commit`, `Black`, `Flake8`/`Ruff`.
* **Tasks:**
  * Configure `.pre-commit-config.yaml` to run linting and formatting automatically on every `git commit`.

---

## 🛠 Phase 2: Modelling & Experimentation

**Goal:** Build a training pipeline that tracks experiments, handles imbalance, and ensures reproducibility.

### 2.1 Model Architecture

* **Tools:** `PyTorch`, `Torchvision`.
* **Script:** `src/models/model.py`
* **Tasks:**
  * Implement Transfer Learning using a pre-trained backbone (e.g., **ResNet50** or **EfficientNet**).
  * Modify the final fully connected layer to output `468` classes.

### 2.2 Configuration Management

* **Tools:** `Hydra`.
* **Tasks:**
  * Create `conf/config.yaml` to decouple hyperparameters from code.
  * Define experiment groups (e.g., `experiment=baseline`, `experiment=weighted_loss`).

### 2.3 Training Loop & Experiment Tracking

* **Tools:** `Weights & Biases (W&B)`, `PyTorch Lightning` (Optional).
* **Script:** `src/train_model.py`
* **Tasks:**
  * **Class Imbalance Strategy:** Calculate and apply **Class Weights** (`sklearn.utils.class_weight`) to the Loss function.
  * **Logging:** log `Training Loss`, `Validation Accuracy`, and `F1-Score` (per class) to W&B.
  * **Checkpointing:** Save the best model based on Validation Loss.

### 2.4 Unit Testing

* **Tools:** `Pytest`.
* **Tasks:**
  * **Data Tests (`tests/test_data.py`):** Verify that no data is leaking between splits and all species exist in the training set.
  * **Model Tests (`tests/test_model.py`):** Verify input tensor shapes (e.g., `[batch, 3, 224, 224]`) and output shapes (`[batch, 468]`).

---

## ☁️ Phase 3: Cloud & Deployment

**Goal:** Containerize the application and deploy it to a scalable cloud environment.

### 3.1 Dockerization

* **Tools:** `Docker`.
* **Tasks:**
  * Create a `Dockerfile` for the training application.
  * **Optimization:** Do *not* copy the dataset into the Docker image. Configure the entrypoint to pull data via DVC upon runtime.
  * Build and test the container locally.

### 3.2 Cloud Training

* **Tools:** `Google Vertex AI` or `GCP Compute Engine`.
* **Tasks:**
  * Submit a training job to the cloud to utilize GPU resources.
  * Verify that logs appear in W&B and checkpoints are saved to Cloud Storage.

### 3.3 Inference API & Deployment

* **Tools:** `FastAPI`, `Google Cloud Run`.
* **Script:** `src/predict_model.py`
* **Tasks:**
  * **API Development:** Create a `/predict` endpoint that accepts an image file and returns the Species Name (mapped from ID).
  * **Deployment:** Deploy the Docker container to **Cloud Run** for serverless inference.

---

## 🛡 Phase 4: Operations & Robustness

**Goal:** Validate model reliability in real-world conditions and automate the pipeline.

### 4.1 Robustness Testing (The "In-Situ" Experiment)

* **Concept:** Domain Drift Evaluation.
* **Tasks:**
  * **Experiment A (Baseline):** Train on random mix of Controlled + In-Situ.
  * **Experiment B (Drift):** Train *only* on Controlled images; Evaluate *only* on In-Situ images.
  * **Reporting:** document the accuracy drop to demonstrate the need for data augmentation or domain adaptation.

### 4.2 CI/CD Pipeline

* **Tools:** `GitHub Actions`.
* **Tasks:**
  * **Continuous Integration:** Trigger `pytest` and `flake8` on every push to `main`.
  * **Continuous Deployment:** Trigger a **Google Cloud Build** to rebuild and push the Docker image whenever the code is updated.

### 4.3 Documentation

* **Tools:** `MkDocs`.
* **Tasks:**
  * Generate a static documentation site in `docs/`.
  * Document the "Fish Species" dataset structure, the cleaning logic, and how to run the API.
