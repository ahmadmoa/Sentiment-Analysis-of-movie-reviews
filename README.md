
# Movie Review Sentiment Analysis Project

This project performs sentiment analysis on movie reviews using multiple machine learning models. The reviews are classified into two categories: positive and negative. It involves preprocessing text data, feature extraction, model training and evaluation, and saving models for future use.

## File Structure

- `main.py`: This file contains the core logic for data processing, model training, and evaluation.
- `test_script.py`: This script is used for testing the saved models by loading the preprocessed data, vectorizers, scalers, and the trained models.
- `final_movie_reviews.pdf`: A document detailing the entire workflow, preprocessing steps, feature extraction, model training (SVM and Logistic Regression), and evaluation metrics.

## Setup

### Requirements

- Python 3.x
- Libraries: pandas, sklearn, nltk, joblib, and other dependencies as required by the scripts.

### Install Required Libraries

```bash
pip install -r requirements.txt
```

Ensure you have the necessary dependencies like nltk, joblib, and scikit-learn.

## Preprocessing

The data undergoes several preprocessing steps before feeding it into the models:
- **HTML Removal**: Strips any HTML tags.
- **Lowercasing**: Converts all text to lowercase.
- **Punctuation Removal**: Removes punctuation marks.
- **Tokenization**: Breaks the text into tokens (words).
- **Stop Word Removal**: Removes common words that don't contribute to sentiment.
- **Stemming**: Reduces words to their root form.
- **Scaling**: Standard and MaxAbs scaling techniques are applied.
- **Dimensionality Reduction**: Reduces feature space using SVD (Singular Value Decomposition).

## Feature Extraction

- **TF-IDF Vectorizer**: Converts text into numerical features based on term frequency and inverse document frequency.

## Models

The project implements two models for sentiment classification:

### Support Vector Machine (SVM)
- **Hyperparameter tuning**: Using GridSearchCV to find the best parameters.
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC are used for evaluating performance.

### Logistic Regression
- **Hyperparameter tuning**: Uses GridSearchCV to optimize C, penalty, and solver.
- **Evaluation**: Similar metrics as SVM.

## Model Training

Models are trained on the preprocessed training data and evaluated on test data.

- **Best Model Selection**: Based on accuracy and performance metrics, the best model is chosen and saved for future use.

## Saving Models

Models and preprocessing components (like vectorizers and scalers) are saved using `joblib` for later use.

## Running the Project

1. **Preprocess and Train Models**: Run `main.py` to preprocess the data, train the models, and evaluate their performance.

```bash
python main.py
```

2. **Test the Models**: Use `test_script.py` to test the saved models on new data and print evaluation metrics.

```bash
python test_script.py
```
The dataset used in this project is from the Cornell University Movie Review Data. You can access the dataset here: [Cornell Movie Review Data](https://www.cs.cornell.edu/people/pabo/movie-review-data/)

## Model Evaluation

- **SVM Model**: Achieves high performance with AUC score of 0.92 and other metrics.
- **Logistic Regression Model**: Compares similarly, but SVM outperforms for this dataset.

## Conclusion

The SVM model is preferred based on accuracy metrics, and the trained models are saved for deployment in real-world applications.
