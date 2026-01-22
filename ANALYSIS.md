# In-Depth Change Analysis

This revision enumerates the touched files with emphasis on the largest behavioral changes. Links use workspace paths.

## Runtime and Core Logic

- [src/group_56/api.py](src/group_56/api.py): Major API uplift (M22, M23, M27, M28). Adds Prometheus metrics
  (request counts, errors, latency, model load), mounts `/metrics`, and introduces `/monitoring` that renders
  Evidently HTML from logged predictions. Prediction handler validates image types, extracts structured features,
  logs each prediction to a local prediction database (M27), and batches GCS uploads every 10 rows. Model loading
  pulls from GCS with gsutil fallback, selects CUDA/MPS/CPU automatically, records metadata in `MODEL_INFO`, and
  exposes `/model/info` and hardened `/health`. Error counters are labeled for observability (M28), and top-k
  responses are returned consistently.
- [src/group_56/data_drift.py](src/group_56/data_drift.py): New Evidently-driven drift module (M27) that loads recent
  predictions, builds reference/current splits, generates DataDrift/DataQuality/TargetDrift reports, saves
  `data_drift_report.html`, and runs test suites for pass/fail signaling. Includes CLI entry to run end-to-end from
  the terminal.
- [src/group_56/extract_features.py](src/group_56/extract_features.py): New image feature extractor (M27) computing
  brightness, contrast, Laplacian sharpness, per-channel mean/std, aspect ratio, and approximate size, enabling
  drift on tabular features derived from images.
- [src/group_56/train.py](src/group_56/train.py): Training pipeline updated for sweep compatibility (M14), improved
  logging, config handling, and artifact persistence to align with new Docker and workflow paths (M10, M21).
- [src/group_56/evaluate.py](src/group_56/evaluate.py): Evaluation path refreshed to consume new artifacts/configs;
  aligns with sweep outputs and logging conventions (M14, M16).
- [src/group_56/model.py](src/group_56/model.py): Model wrapper adjusted for new loading/saving conventions and
  metadata exposure; supports downstream API usage (M22, M23).
- [src/group_56/sweep_agent.py](src/group_56/sweep_agent.py): New sweep runner coordinating W&B sweeps for
  hyperparameter search (M14), bridging configs and train/eval routines.
- [src/group_56/data.py](src/group_56/data.py): Data utilities refined to support training/eval data flows consistent
  with new features; minor robustness tweaks (M6, M16).
- [src/group_56/convert_txt_to_csv.py](src/group_56/convert_txt_to_csv.py): Data prep helper tweaked to produce
  consistent CSV inputs for downstream pipelines (M6).

## Monitoring, Alerting, and Ops

- [scripts/setup_gcp_monitoring.sh](scripts/setup_gcp_monitoring.sh): Automates GCP Monitoring setup (M28). Creates an
  email channel and three alert policies (error rate, latency, unresponsive service) with validation commands and
  manual fallbacks.
- Prometheus and drift wiring: Metrics are exposed via `/metrics` for Cloud Run scraping; `/monitoring` reuses
  prediction logs to compute drift and serve interactive HTML.

## CI/CD and Build

- [.github/workflows/deploy-cloud-run.yaml](.github/workflows/deploy-cloud-run.yaml): New Cloud Run deploy pipeline
  (M23) building/pushing the API image, deploying, and running smoke tests. Supports workload identity federation or
  JSON key auth with guarded steps.
- [.github/workflows/data-change.yaml](.github/workflows/data-change.yaml): New workflow triggered on data updates to
  retrain/evaluate and keep artifacts fresh (M19, M21).
- [.github/workflows/model-registry.yaml](.github/workflows/model-registry.yaml): New workflow for model promotion and
  registry interactions (M19).
- [.github/workflows/tests.yaml](.github/workflows/tests.yaml): Test workflow expanded to cover new dependencies and
  jobs (M17).
- [.github/workflows/pre-commit-update.yaml](.github/workflows/pre-commit-update.yaml): Scheduling/steps adjusted to
  refresh hooks automatically (M18).
- [cloudbuild.yaml](cloudbuild.yaml): Root Cloud Build builds and pushes train/eval images (M21); duplicate under
  reports removed.
- [reports/cloudbuild.yaml](reports/cloudbuild.yaml): Deleted duplicate to avoid confusion.

## Container Images

- [dockerfiles/api.dockerfile](dockerfiles/api.dockerfile): API image hardened with proper deps, caching, runtime
  envs, and sizing optimizations for Cloud Run (M10, M21, M23).
- [dockerfiles/train.dockerfile](dockerfiles/train.dockerfile): Training image refreshed to include new deps and
  alignment with sweep/training entrypoints (M10, M21).
- [dockerfiles/evaluate.dockerfile](dockerfiles/evaluate.dockerfile): Eval image updated to mirror train/runtime
  dependency set and entrypoints (M10, M21).

## Documentation (MkDocs and Guides)

- [docs/source/CLOUD_RUN_DEPLOYMENT.md](docs/source/CLOUD_RUN_DEPLOYMENT.md): End-to-end Cloud Run deployment guide
  with validated steps and outcomes (M23, M32).
- [docs/source/MONITORING.md](docs/source/MONITORING.md): Monitoring guide covering metrics exposure, scraping, and
  alerting (M28, M32).
- [docs/source/DATA_DRIFT_DETECTION.md](docs/source/DATA_DRIFT_DETECTION.md): Drift detection design, feature set, and
  Evidently usage flow (M27, M32).
- [docs/source/LOAD_TESTING.md](docs/source/LOAD_TESTING.md): Load test procedure, execution, and interpretation
  (M24, M32).
- [docs/source/WEEK2_SUMMARY.md](docs/source/WEEK2_SUMMARY.md): Weekly wrap-up capturing progress and outcomes (M32).
- [docs/API_USAGE.md](docs/API_USAGE.md): API calling patterns and examples (M22, M32).
- [docs/LOGGING_AND_SWEEPS.md](docs/LOGGING_AND_SWEEPS.md): Guidance on logging strategy and sweep execution (M14,
  M32).
- [docs/mkdocs.yaml](docs/mkdocs.yaml): Navigation updated to surface all new docs (M32).

## Reports and Figures

- [reports/README.md](reports/README.md): Report template completed with narrative and figure references for
  evaluation (M32).
- [reports/report.py](reports/report.py): Helper that validates report content lengths, enforces image-count limits,
  and renders HTML (M32).
- [reports/figures/bucket.png](reports/figures/bucket.png), [reports/figures/build.png](reports/figures/build.png),
  [reports/figures/overview.png](reports/figures/overview.png),
  [reports/figures/registry.png](reports/figures/registry.png), [reports/figures/wandb.png](reports/figures/wandb.png):
  New diagrams showing bucket, build, system overview, registry, and W&B views to accompany the report (M32).

## Testing and Quality

- [tests/test_api.py](tests/test_api.py): Expanded to cover new endpoints, health/info, and error paths; ensures
  metrics and prediction flows behave (M16, M24).
- [tests/test_data.py](tests/test_data.py): Updated to reflect revised data utilities and loaders (M16).
- [tests/test_model.py](tests/test_model.py): Adjusted for new model interfaces and artifacts (M16).
- [tests/load_test.py](tests/load_test.py): New load-testing script to stress the API and validate latency/error
  thresholds (M24).
- [tests/__init__.py](tests/__init__.py): Init updated to support the broadened test suite (M16).

## Configuration and Dependencies

- [pyproject.toml](pyproject.toml): Dependency and project metadata aligned with monitoring, drift, and sweep features
  (M2, M14, M27, M28).
- [requirements.txt](requirements.txt): Runtime deps expanded (Evidently, Prometheus client, scikit-learn, loguru,
  etc.) (M2, M27, M28).
- [requirements_dev.txt](requirements_dev.txt): Dev stack synced with lint/test tooling updates (M2, M17, M18).
- [requirements_tests.txt](requirements_tests.txt): New test-only dependency list for CI isolation (M16, M17).
- [uv.lock](uv.lock): Lockfile regenerated to capture the expanded dependency graph (M2).
- [.pre-commit-config.yaml](.pre-commit-config.yaml): Hook versions bumped; formatting/linting tightened (M18).
- [configs/sweep_config.yaml](configs/sweep_config.yaml): New sweep configuration for W&B hyperparameter searches
  (M14).
- [tasks.py](tasks.py): Invoke tasks updated to expose new actions (tests, docs, sweeps) and align with current
  scripts (M9, M17, M32).

## Assets and Artifacts

- [models/quick_deploy/fish_classifier.pt](models/quick_deploy/fish_classifier.pt): Added quick-deploy model artifact
  for immediate serving (M23).
- [reports/figures/*](reports/figures/overview.png): New figures set used by the report and docs (M32).

## CI Housekeeping and Ignore Rules

- [.gitignore](.gitignore): Expanded to ignore generated reports, caches, and data logs tied to the new workflows
  (M17, M21, M24, M27).
- [Project Checklist.md](Project%20Checklist.md): Progress checklist updated for completed M27/M28 tasks and
  documentation.

## Licensing and Legal

- [LICENSE](LICENSE): Populated with MIT license text.

## Deleted or Removed

- [reports/cloudbuild.yaml](reports/cloudbuild.yaml): Removed to avoid duplicate Cloud Build definitions (root file is the source of truth).

## Behavioral Impact Snapshot

- Observability: Prometheus metrics, drift HTML, and alert setup script bring production-grade monitoring.
- Reliability: Deploy workflow now health-checks post-deploy and supports safer auth selection (WIF or JSON key).
- Reproducibility: Deployment, monitoring, drift, and load testing are documented; CI and images are aligned with code.
- Compliance and reporting: Report template, figures, and license updates make the project ready for evaluation.

## Quick Navigation

- Runtime focus: [src/group_56/api.py](src/group_56/api.py) plus drift chain
  [src/group_56/extract_features.py](src/group_56/extract_features.py) → prediction database CSV →
  [src/group_56/data_drift.py](src/group_56/data_drift.py) → `/monitoring`.
- Deploy focus: [.github/workflows/deploy-cloud-run.yaml](.github/workflows/deploy-cloud-run.yaml),
  [dockerfiles/api.dockerfile](dockerfiles/api.dockerfile), [cloudbuild.yaml](cloudbuild.yaml).
- Monitoring docs: [docs/source/MONITORING.md](docs/source/MONITORING.md),
  [scripts/setup_gcp_monitoring.sh](scripts/setup_gcp_monitoring.sh).
- Drift docs: [docs/source/DATA_DRIFT_DETECTION.md](docs/source/DATA_DRIFT_DETECTION.md).
- Report assets: [reports/README.md](reports/README.md) and [reports/figures/overview.png](reports/figures/overview.png).
