# Intracranial Hemorrhage Detection

## Phase 1: CNN + Attention 
https://github.com/YunanWu2168/SA-MIL

## Phase 2: SVGP + Attention

## Description

## Phase 2: SVGP + Attention

This repository contains the code for Phase 2 of the Intracranial Hemorrhage Detection project, building upon the CNN + Attention model from Phase 1. This phase introduces a Sparse Variational Gaussian Process (SVGP) layer combined with an attention mechanism for Multiple Instance Learning (MIL). We extend the Phase 1 model by self-implementing the SVGP layer and integrating it with the existing attention mechanism to create a multi-label multi-instance learning framework. This allows the model to effectively handle datasets where each sample consists of multiple instances, and each instance can have multiple labels. We inherit structure of model and data representation from phase 1.


## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features

*   **Train an SVGP-MIL model for intracranial hemorrhage detection:**  This project implements a Sparse Variational Gaussian Process (SVGP) combined with an attention mechanism for Multiple Instance Learning (MIL) to classify intracranial hemorrhage from medical imaging data.
*   **Perform multi-label classification:** The model can predict the presence of multiple types of intracranial hemorrhage (e.g., epidural, subdural, subarachnoid) in a single patient case.
*   **Leverage pre-extracted features:** The code works with pre-extracted image features, allowing users to quickly experiment with the model without processing large image datasets themselves.
*   **Experiment with data redundancy:** The repository provides the ability to train and evaluate both a standard SVGP model and a redundancy-aware (R-SVGP) model, enabling investigation into the impact of data redundancy on performance.
*   **Evaluate model robustness through independent runs:**  The dataset is structured to enable 5 independent training and testing runs, allowing for a robust assessment of the model's generalization capabilities.
*   **Visualize attention weights:** The code includes functionality to visualize the attention weights learned by the model, providing insights into which instances are most relevant for the classification decision and helps to see the importance of each slice.
*   **Test generalization on the CQ500 dataset:** Users can evaluate the model's ability to generalize to a different dataset (CQ500) to see how well the model transfers to new patient populations.

## Getting Started

### Prerequisites

To run this project, you'll need the following:

*   Python (3.10 or higher)
    *   It's highly recommended to use a virtual environment (venv or conda) to manage dependencies.
*   pip (Python package installer)
*   PyTorch (2.4.1)
    *   Installation instructions can be found on the [PyTorch website](https://pytorch.org/get-started/locally/). Make sure to select the appropriate configuration for your operating system and hardware (CPU/GPU).

### Installation

Step-by-step instructions on how to install the project.

1.  Clone the repository:
    ```
    git clone https://github.com/nhatminhtrieu/Brain-Stroke-Diagnosis.git
    ```
2.  Navigate to the project directory:
    ```
    cd Brain-Stroke-Diagnosis
    ```
3.  Install dependencies (example using pip):
    ```
    pip install -r requirements.txt
    ```

## Usage

This section details how to run the code and understand the dataset used in this project.

### 1. Dataset Acquisition (Optional)

For plotting and visualization purposes, you might want to download the image data associated with the RSNA Intracranial Hemorrhage Detection competition. The dataset can be found on Kaggle:

[https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data)

**Note:** The core code functionality (training and evaluation) *does not* require the image data. It relies on pre-extracted features provided in the CSV files.  If you are only interested in running the model and analyzing the metrics, you can skip this step.

### 2. Running the Code

The main script (`SVGPMIL.ipynb`) can be executed directly. Ensure that you have the necessary dependencies installed (see [Prerequisites](#prerequisites)).

1.  **Data Location:**  Place the provided CSV files (`rsna_train_*_redundancy.csv` and `rsna_test_*_redundancy.csv` or `cq500_test_*_redundancy.csv`) in a directory named `data/rsna/` relative to the script's location.  If you choose to use a different directory, you will need to modify the file paths in the main section of the code.
2.  **Execution:**  Run the `SVGPMIL.ipynb` notebook using Jupyter or a similar environment.

### 3. Understanding Data Variants

The `data/rsna/` folder contains several CSV files. These files represent two model variations and five independent runs:

*   **Model Variations:**

    *   Files with the suffix `_update.csv`: These files are designed for training and testing the **SVGP** model *without* data redundancy.
    *   Files with the suffix `_redundancy.csv`: These files are designed for training and testing the **R-SVGP** model, which *incorporates* data redundancy.
*   **Independent Runs:**

    *   `rsna_train_0_*.csv`, `rsna_train_1_*.csv`, ..., `rsna_train_4_*.csv`: Each of these files contains a different shuffling of data and data split, enabling 5 independent training runs to evaluate robustness.
    *   `rsna_test_0_*.csv`, `rsna_test_1_*.csv`, ..., `rsna_test_4_*.csv`: Corresponding test files for each training split.
*   **Dataset Option:**

    *   `cq500_test_*.csv`: These can be used to test the generalization capability

### 4. Interpreting Results

After running the code, you will get metrics such as accuracy, precision, and so on. Moreover, the Attention weights visulization.



## License

This project is licensed under the [License Name] License - see the [LICENSE.txt](LICENSE.txt) file for details. 



