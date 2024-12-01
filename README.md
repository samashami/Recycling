
# Trashnet Continual Learning for Recycling

This project implements a machine learning model for recycling using **Trashnet** and **Continual Learning** techniques. The goal is to classify different types of trash items (e.g., paper, plastic, metal) and improve the model's performance over time by introducing new categories while preserving previous knowledge.

## Project Overview

The project utilizes the Trashnet dataset to train a model that can classify waste items into categories. It integrates **Continual Learning** (CL) techniques to handle new classes introduced over time, preventing the model from forgetting previously learned classes. This is crucial for building adaptive systems in dynamic environments, such as recycling processes.

## Features

- **Trashnet Dataset**: A publicly available dataset with images of different types of trash (paper, plastic, metal, etc.).
- **Continual Learning**: Techniques to handle learning new categories without forgetting old ones (e.g., using methods like **Elastic Weight Consolidation (EWC)** or **Replay-based** methods).
- **Data Augmentation**: To improve the model's robustness, several data augmentation techniques are applied (e.g., rotation, flipping, cropping).
- **Model Evaluation**: Evaluation of model performance over time, including accuracy, precision, and recall for each category.

## Installation

### Prerequisites

To run the project, you need the following:

- Python 3.6+
- pip (Python package installer)
- TensorFlow or PyTorch (depending on your preference)
- Other dependencies listed in `requirements.txt`

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/trashnet-continual-learning.git
cd trashnet-continual-learning
```

### Step 2: Install dependencies

You can install the required Python libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 3: Download the Trashnet dataset

The Trashnet dataset can be downloaded from its official source or can be preprocessed from the raw images (please refer to the dataset's documentation for download links).

After downloading, place the dataset in the `data/` directory.

## Usage

### Training the Model

To train the model from scratch, run the following command:

```bash
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

This will train the model using the default parameters. You can adjust the arguments as needed.

### Evaluating the Model

Once the model is trained, you can evaluate it on the validation dataset by running:

```bash
python evaluate.py --model_path path_to_trained_model
```

This will output the model’s accuracy, precision, recall, and F1-score for each class.

### Continual Learning Setup

To train the model with Continual Learning techniques, use the following command:

```bash
python continual_learning.py --method EWC --epochs 50 --batch_size 32 --learning_rate 0.001
```

Replace `--method` with the continual learning strategy of your choice (e.g., EWC, replay-based methods).

## Continual Learning Methods Implemented

- **Elastic Weight Consolidation (EWC)**: Helps mitigate catastrophic forgetting by adding a regularization term based on the importance of weights.
- **Replay-based Methods**: Randomly selects a subset of data from previous tasks and uses it to fine-tune the model while learning new classes.
- **Progressive Neural Networks**: (optional) Involves adding new columns to the network as new tasks are introduced.

## Folder Structure

```
trashnet-continual-learning/
│
├── data/                # Contains dataset and preprocessing scripts
│   └── raw_images/      # Raw Trashnet images
│
├── models/              # Stores trained models
├── src/                 # Contains training, evaluation, and CL code
│   ├── train.py         # Main training script
│   ├── evaluate.py      # Model evaluation script
│   └── continual_learning.py  # Implements Continual Learning methods
│
├── requirements.txt     # List of dependencies
└── README.md            # This file
```

## Contributing

Contributions are welcome! If you want to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Trashnet Dataset](https://github.com/garythung/trashnet) - for the dataset used in this project.
- [Continual Learning Resources](https://arxiv.org/abs/1810.12488) - for various techniques in continual learning.
- [TensorFlow/PyTorch](https://www.tensorflow.org/) - for the deep learning framework.

---
