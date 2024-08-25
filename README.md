# Ensemble Regressor and Classifier

This repository contains the implementation of two ensemble learning techniques: a **Random Forest Regressor** for the Boston housing dataset and a custom **AdaBoost Classifier** for the Breast Cancer dataset. The code also compares the custom AdaBoost implementation with scikit-learn’s built-in AdaBoost classifier.

## Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Model 1: Random Forest Regression (Boston Housing Dataset)](#model-1-random-forest-regression-boston-housing-dataset)
  - [Model Training and Hyperparameter Tuning](#model-training-and-hyperparameter-tuning)
  - [Evaluation](#evaluation)
- [Model 2: AdaBoost Classification (Breast Cancer Dataset)](#model-2-adaboost-classification-breast-cancer-dataset)
  - [Custom AdaBoost Classifier Implementation](#custom-adaboost-classifier-implementation)
  - [Model Training and Hyperparameter Tuning](#model-training-and-hyperparameter-tuning-1)
  - [Evaluation](#evaluation-1)
  - [Comparison with Scikit-Learn's Implementation](#comparison-with-scikit-learns-implementation)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [References](#references)

## Dataset Preparation

1. **Boston Housing Dataset**  
   The dataset is fetched from the original source and processed for Random Forest Regression. The features and target variable are combined and shuffled before splitting into training, validation, and testing sets.

2. **Breast Cancer Dataset**  
   The dataset is loaded from scikit-learn's `load_breast_cancer()` function. It is split into training, validation, and testing sets for classification.

## Model 1: Random Forest Regression (Boston Housing Dataset)

### Model Training and Hyperparameter Tuning

The Random Forest Regressor is trained using `GridSearchCV` to identify the best hyperparameters. The hyperparameter grid includes:
- `n_estimators`: [50, 100, 150]
- `max_depth`: [None, 5, 10, 15]
- `max_features`: [log2 of feature count]

### Evaluation

The model is evaluated using Mean Squared Error (MSE) and a scatter plot of actual vs. predicted values is displayed.

## Model 2: AdaBoost Classification (Breast Cancer Dataset)

### Custom AdaBoost Classifier Implementation

An AdaBoost classifier is implemented from scratch using a decision tree (with a max depth of 1) as the weak learner. The algorithm iteratively adjusts the weights of the training samples and fits subsequent weak learners.

### Model Training and Hyperparameter Tuning

The model is trained with different numbers of estimators ([50, 100, 150]) to determine the best performing model based on validation accuracy.

### Evaluation

The best custom model is evaluated using classification metrics including:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Comparison with Scikit-Learn's Implementation

The custom AdaBoost implementation is compared with scikit-learn’s AdaBoost implementation using `GridSearchCV` to identify the best number of estimators.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/satwika787/ensemble-regressor-classifier.git

## Dependencies
The project requires the following libraries:

Python 3.x
scikit-learn
numpy
pandas
matplotlib
pip install -r requirements.txt

## References
Boston Housing Dataset: [Original Source]
Breast Cancer Dataset: [Scikit-learn Datasets]
