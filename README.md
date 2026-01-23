# Machine Learning Operation Project Description- Image Classification on fish species dataset.

Group 56:

Alessandra Carrara s252976
Emmanouil Tsillingeridis s253154
Guido Furlanetto s243131
Nikolaos Fergadakis s253129

## Project Goal

The overall goal of this project is to design, implement, and rigorously evaluate an end-to-end machine learning pipeline for image-based fish species classification. The task focuses on automatically identifying the species of a fish from an input image, a problem with practical relevance in marine biology, fisheries management, environmental monitoring, and automated underwater analysis systems.

Beyond achieving strong predictive performance, the project emphasizes reproducibility, structured experimentation, and scalability by following a principled MLOps-oriented workflow. This includes systematic data versioning, configuration-driven experimentation, model tracking, and deployment-ready practices, ensuring that results are reliable, transparent, and easily reproducible.

## Data

For this project, we used the Fish Species Classification dataset, which consists of approximately 4000 labeled RGB images distributed across ~450 different fish species. The dataset consists of 3 folder of images (cropped,raw,numbered), and one .txt file which gives us more details about the labels of the images. This project addresses a fine-grained image classification problem, where the goal is to differentiate between visually similar fish species. In such settings, high-quality, discriminative visual features are critical, and irrelevant background information can negatively affect model performance.

For this reason, we exclusively use the cropped images provided with the dataset. These images contain only the fish itself, with background regions removed through manual cropping. Since this is a classification task—not an object detection task—the presence of background pixels does not provide additional useful information and may instead introduce noise.

The images depict fish under diverse real-world conditions, including variations in:

- lighting
- orientation and pose
- scale

These factors introduce both intra-class variability and inter-class similarity, making the classification task non-trivial.

The data modality consists of static color images. All images will be resized to a fixed resolution suitable for convolutional neural networks (e.g., 224×224 RGB pixels). The dataset will be split into training, validation, and test sets, and these splits will be kept consistent across all experiments to ensure fair and reproducible comparisons.

## Models and Methodology

To benchmark performance on the fish species classification task, we evaluate convolutional neural networks based on transfer learning. Specifically, we train and assess pretrained ResNet architectures, including ResNet-18, ResNet-50 and/or ResNet-34. These models are widely used for image classification and are known to generalize well, particularly in settings where the available training data is limited.

All models are initialized with weights pretrained on ImageNet and fine-tuned on our dataset. To ensure compatibility with these pretrained weights, we apply the official torchvision preprocessing pipeline for ResNet models, which is aligned with ImageNet statistics. This preprocessing strategy is well suited for transfer learning and helps maintain consistency between the training data and the distribution on which the models were originally trained.

Note: Running dvc pull for the first time will open a browser tab for Google authentication. Please sign in with any Google account to access the public data folder."

## Project flowchart

![Demo image](report/figures/mlops_architecture.png)

## How to install

Installing the project on your machine should be straighforward although Pytorch Geometric can cause some trouble. Clone the repo:

```
git clone https://github.com/FergadakisN/group56.git
```

Install requirements, preferably in seperate virtual environment:

```
pip install -r requirements.txt
```

## How to run

Running the training locally:

```
invoke train
```

Evaluate the model:

```
invoke evaluate
```

## Project structure

The directory structure of the project looks like this:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
|   ├── evaluate.dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yaml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── group_56/
│   │   ├── __init__.py
│   │   ├── api.py
|   |   ├── convert_txt_to_csv.py
│   │   ├── data_drift.py
│   |   ├── data.py
│   │   ├── evaluate.py
│   │   ├── extract_features.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── sweep_agent.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── load_test.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

## Overall project checklist

The checklist is _exhaustive_ which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

- [x] Create a git repository (M5)
- [x] Make sure that all team members have write access to the GitHub repository (M5)
- [x] Create a dedicated environment for you project to keep track of your packages (M2)
- [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
- [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
- [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
- [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
      `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
- [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
- [x] Do a bit of code typing and remember to document essential parts of your code (M7)
- [x] Setup version control for your data or part of your data (M8)
- [x] Add command line interfaces and project commands to your code where it makes sense (M9)
- [x] Construct one or multiple docker files for your code (M10)
- [x] Build the docker files locally and make sure they work as intended (M10)
- [x] Write one or multiple configurations files for your experiments (M11)
- [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
- [ ] Use profiling to optimize your code (M12)
- [x] Use logging to log important events in your code (M14)
- [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
- [x] Consider running a hyperparameter optimization sweep (M14)
- [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

- [x] Write unit tests related to the data part of your code (M16)
- [x] Write unit tests related to model construction and or model training (M16)
- [x] Calculate the code coverage (M16)
- [x] Get some continuous integration running on the GitHub repository (M17)
- [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
- [x] Add a linting step to your continuous integration (M17)
- [x] Add pre-commit hooks to your version control setup (M18)
- [x] Add a continues workflow that triggers when data changes (M19)
- [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
- [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
- [x] Create a trigger workflow for automatically building your docker images (M21)
- [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
- [x] Create a FastAPI application that can do inference using your model (M22)
- [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
- [x] Write API tests for your application and setup continues integration for these (M24)
- [x] Load test your application (M24)
- [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
- [ ] Create a frontend for your API (M26)

### Week 3

- [x] Check how robust your model is towards data drifting (M27)
- [x] Setup collection of input-output data from your deployed application (M27)
- [x] Deploy to the cloud a drift detection API (M27)
- [x] Instrument your API with a couple of system metrics (M28)
- [x] Setup cloud monitoring of your instrumented application (M28)
- [x] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
- [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
- [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
- [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

- [x] Write some documentation for your application (M32)
- [ ] Publish the documentation to GitHub Pages (M32)
- [x] Revisit your initial project description. Did the project turn out as you wanted?
- [x] Create an architectural diagram over your MLOps pipeline
- [x] Make sure all group members have an understanding about all parts of the project
- [x] Uploaded all your code to GitHub

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
