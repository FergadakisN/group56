# group_56

A machine learning operations project for fish species classification using deep learning and MLOps best practices.

## Project Goal

The overall goal of this project is to design, implement, and evaluate a complete machine learning pipeline for image-based fish species classification. The task focuses on automatically identifying the species of a fish given an input image, a problem that is relevant for applications in marine biology, fisheries management, environmental monitoring, and automated underwater analysis systems.

Beyond raw model performance, this project also aims to explore best practices in model comparison, reproducibility, and structured experimentation, following a principled MLOps-oriented workflow.

## Dataset Description

For this project, we will initially use the Fish Species Classification dataset, which consists of approximately 9,000 labeled RGB images distributed across 30 different fish species. Each class contains a few hundred images, making the dataset moderately sized but still challenging due to class variability.

The images depict fish under diverse real-world conditions, including variations in:

- lighting
- background clutter
- orientation and pose
- scale

These factors introduce both intra-class variability and inter-class similarity, making the classification task non-trivial.

The data modality consists of static color images. All images will be resized to a fixed resolution suitable for convolutional neural networks (e.g., 224×224 pixels). The dataset will be split into training, validation, and test sets, and these splits will be kept consistent across all experiments to ensure fair and reproducible comparisons.

## Models and Methodology

To benchmark performance on the fish species classification task, we will evaluate a combination of baseline models and transfer-learning-based convolutional neural networks.

We will begin by training a Baseline CNN model from scratch, consisting of:

- 2–4 convolutional blocks
- pooling layers
- a lightweight fully connected classifier head

This baseline serves as a sanity check and provides a reference point for understanding the difficulty of the task and the benefits of more advanced architectures.

In addition, we will train and evaluate pretrained ResNet architectures, such as ResNet-18 and ResNet-50, using transfer learning. These models are well-established in image classification tasks and are known to generalize well even when the available dataset is relatively limited. We expect the ResNet-based models to significantly outperform the baseline CNN in terms of accuracy and robustness.

All models will be evaluated using the same metrics (e.g., accuracy, precision, recall, and F1-score) and identical data splits to ensure a controlled and meaningful comparison.

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
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
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

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
