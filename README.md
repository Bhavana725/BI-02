Titanic Survival Prediction
This repository contains a comprehensive guide to building and improving predictive models to determine the likelihood of survival for passengers on the Titanic. The project employs various data science techniques and machine learning algorithms, including Random Forest, Support Vector Machine (SVM), Logistic Regression, and Gradient Boosting. Additionally, it explores methods to enhance model performance through  hyperparameter tuning, handling class imbalance, and cross validation.

Project Overview
The sinking of the Titanic is one of the most infamous shipwrecks in history. One of the main reasons it resulted in such loss of life was that there were not enough lifeboats for the passengers and crew. While there was some element of luck involved in surviving the sinking, it seems some groups of people were more likely to survive than others.

This project aims to predict whether a given passenger on the Titanic would have survived or not based on various features such as age, gender, passenger class, etc.

Steps and Techniques
1. Data Loading and Exploration
The dataset is loaded using Pandas, and initial exploration is performed to understand the structure and contents of the data.

2. Data Cleaning
Missing values are handled using techniques such as median imputation for numerical features and mode imputation for categorical features. Unnecessary columns are dropped to simplify the dataset.

3. Data Preprocessing
Categorical variables are encoded using Label Encoding. Numerical features are scaled using Standard Scaler to ensure they have a mean of 0 and a standard deviation of 1.

4. Handling Class Imbalance
The Synthetic Minority Over-sampling Technique (SMOTE) is used to handle class imbalance in the training data.

5. Model Building and Training
Four different machine learning models are trained:

Random Forest Classifier
Logistic Regression
Support Vector Machine (SVM)
Gradient Boosting Classifier
6. Hyperparameter Tuning
Grid Search is used to find the best hyperparameters for the Random Forest model.

7. Cross-validation
Cross-validation is performed to evaluate the models and ensure they perform well on unseen data.


8. Evaluation
The models are evaluated using accuracy score, confusion matrix, and classification report. Confusion matrices for each model are plotted for better visualization.

How to Run the Code
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook:

bash
Copy code
jupyter notebook titanic_survival_prediction.ipynb
Results
The accuracy of each model along with their confusion matrices and classification reports are provided. The final combined Voting Classifier model demonstrates improved performance by leveraging the strengths of individual models.

Conclusion
This project showcases the application of various data science techniques and machine learning algorithms to solve a binary classification problem. Through feature engineering, hyperparameter tuning, the performance of the models is significantly enhanced.
