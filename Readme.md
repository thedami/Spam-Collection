# Spam Detection using Supervised Learning

## Project Overview

This project implements a spam detection system using various supervised learning techniques. The goal is to accurately classify email messages as either spam or ham (non-spam) based on their content and characteristics.

## Dataset

The project uses the Spam Collection Dataset, which consists of 5,572 SMS labeled messages that have been collected for SMS Spam research. The dataset is available via the UCI Machine Learning Repository.

## Key Features

- Data preprocessing and text cleaning
- Feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency)
- Implementation and comparison of multiple machine learning models:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- Model performance evaluation and comparison

## Methodology

1. **Data Loading and Preprocessing**: 
   - Loaded the SMS spam collection dataset
   - Performed basic text cleaning (lowercasing, punctuation removal)
   - Split the data into training and testing sets

2. **Feature Extraction**:
   - Used TF-IDF vectorization to convert text data into numerical features

3. **Model Development and Training**:
   - Implemented Logistic Regression, Naive Bayes, SVM, and Random Forest classifiers
   - Trained each model on the vectorized training data

4. **Model Evaluation**:
   - Evaluated models using accuracy, precision, recall, and F1-score
   - Analyzed confusion matrices to understand model performance

## Technologies Used

- Python
- Pandas for data manipulation
- Scikit-learn for machine learning models and text vectorization
- NLTK for text preprocessing
- Matplotlib & Seaborn for data visualization
- Jupyter Notebook for development and presentation

## Results

- Achieved high accuracy in spam detection across multiple models
- Compared performance of different classifiers:
  - Logistic Regression: 96.49% accuracy
  - Naive Bayes: 96.85% accuracy
  - SVM: 98.38% accuracy
  - Random Forest: 97.66% accuracy

## Future Improvements

- Experiment with more advanced text preprocessing techniques
- Implement deep learning models like LSTM or BERT for comparison
- Develop a user interface for real-time spam detection
- Expand the dataset to include more recent spam patterns
