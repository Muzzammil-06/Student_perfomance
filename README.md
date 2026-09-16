# Student Performance Prediction

A machine learning project that predicts whether a student will pass or fail based on academic, demographic, and lifestyle-related features from the UCI Student Performance dataset.

The project implements **Logistic Regression from scratch using NumPy**, including data preprocessing, feature normalization, gradient descent optimization, and model evaluation.

## Project Overview

The goal of this project is to use student information to predict a binary outcome:

- `1` → Pass
- `0` → Fail

The model is trained using Logistic Regression implemented manually rather than using a machine learning library such as Scikit-learn.

## Dataset

This project uses the **Student Performance Dataset** from the UCI Machine Learning Repository.

The dataset contains information about students, including:

- Age
- Mother's education
- Father's education
- Travel time
- Study time
- Number of previous failures
- Family relationship quality
- Free time
- Going out
- Alcohol consumption
- Health
- Absences
- Final grade (`G3`)

The target variable is created from the final grade:

```python
pass = (G3 >= 10)
