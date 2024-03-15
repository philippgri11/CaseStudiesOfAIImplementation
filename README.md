# README for Case Studies of AI Implementation Repository

This repository contains the code and documentation for the project "Case Studies of AI Implementation", focusing on the application of machine learning models to predict load demands in industrial settings. The goal is to enhance the accuracy of daily electric load forecasting by integrating various influencing variables such as holiday periods, temperature, and radiation levels.

## Overview

The project demonstrates the implementation and comparison of several machine learning models, including XGBoost, ARD, KNN, and LSTM, to predict load demands with high accuracy. It explores the use of XGBoost for its superior performance and compares it against other models as benchmarks.

## Structure

The repository is structured to include code for preprocessing, model training, hyperparameter optimization, and evaluation. It also contains Jupyter notebooks for Exploratory Data Analysis (EDA) and testing purposes. Documentation is provided to guide users through the setup, execution, and understanding of the project's objectives and outcomes.

### Directory Structure

- `src/`: Contains all Python scripts and modules for data preprocessing, model training, hyperparameter tuning, and evaluation.
- `notebooks/`: Jupyter notebooks for EDA and testing the models with visualizations and insights.
- `data/`: Directory for storing the datasets used in the project.
- `docs/`: Additional documentation.
- `requirements.txt`: A list of Python packages required to run the code in this repository.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository:**
   Clone this repository to your local machine using `git clone https://github.com/philippgri11/CaseStudiesOfAIImplementation.git`.

2. **Install Dependencies:**
   Navigate to the cloned repository directory and install the required Python packages using `pip install -r requirements.txt`.

3. **Explore Notebooks:**
   Launch Jupyter Notebook or JupyterLab and open the notebooks in the `notebooks/` directory to explore the EDA and model testing processes.

4. **Run Scripts:**
   Execute the Python scripts in the `src/` directory to preprocess data, train models, and evaluate their performance. Scripts can be run from the command line or integrated into other Python projects.

## Results and Insights

The main findings from this project underscore the effectiveness of the XGBoost model in predicting load demands, significantly outperforming other models in accuracy and efficiency.
The load forecasts are documented in result as csv.

## Acknowledgments

Special thanks to Siemens AG for their support and collaboration in providing the load data necessary for this research project.
