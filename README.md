Financial Market News Classification:


This project focuses on classifying financial market news articles using a Random Forest classifier. The dataset contains financial news articles with associated labels, and the goal is to build a model that can predict the label of new articles based on their content.

Project Overview:

The project involves the following key steps:

* Data Loading and Exploration: Load and explore the dataset to understand its structure and contents.
* Feature Extraction: Convert the news text into numerical features using Bag of Words.
* Model Training and Evaluation: Train a Random Forest classifier and evaluate its performance on test data.
Dataset
* The dataset used in this project is available at the following URL: Financial Market News Dataset. The dataset contains financial news articles with labels indicating their classification.

Project Steps:

1. Data Loading and Exploration:
  * The dataset is loaded into a Pandas DataFrame from the provided URL.
  * Basic information about the dataset is retrieved, including the first few rows, data types, shape, and column names.
  * The iloc method is used to view specific rows and columns, providing insights into the data structure.

2. Feature Extraction:  
  * Text Conversion to Bag of Words: The CountVectorizer from Scikit-Learn is used to convert text data into numerical features. The ngram_range parameter is set to (1,1) to consider single words (unigrams).
  * Feature Selection: News articles are transformed into a feature matrix using Bag of Words.

3. Train-Test Split:
  * The dataset is split into training and test sets using train_test_split from Scikit-Learn. The split is stratified based on the labels to maintain the proportion of classes in both sets.

4. Model Training and Evaluation:
  * Random Forest Classifier: A Random Forest classifier with 200 estimators is trained on the training data.
  * Prediction and Evaluation: The model’s performance is evaluated on the test data using various metrics, including confusion matrix, classification report, and accuracy score.

Key Functions:

* pandas.read_csv: Loads the dataset into a DataFrame.
* sklearn.feature_extraction.text.CountVectorizer: Converts text data into a Bag of Words representation.
* sklearn.model_selection.train_test_split: Splits the dataset into training and test sets.
* sklearn.ensemble.RandomForestClassifier: Trains the Random Forest model.
* sklearn.metrics: Evaluates the model performance using confusion matrix, classification report, and accuracy score.

Usage:

* Install Dependencies: Ensure that the required libraries are installed. You can install them using pip:
pip install pandas numpy scikit-learn
* Run the Code: Execute the provided code to load the dataset, perform feature extraction, train the model, and evaluate its performance.
* Evaluate Results: Review the output metrics to assess the model’s accuracy and classification performance.

Notes:
* Make sure to replace placeholder code with actual implementation as needed.
* Ensure that the dataset URL and column names in the code match the actual dataset used.
