Customer Churn Prediction Using Machine Learning

Author

Rajesh Kumar
Computer Science Engineering Graduate
Aspiring Data Scientist | Machine Learning Engineer

GitHub: https://github.com/Rajeshkumar0707

Project Overview

Customer churn is an important business problem where companies need to identify customers who are likely to stop using their services.

This project develops a Machine Learning-based Customer Churn Prediction system using customer information such as age, gender, tenure, and monthly charges.

The project covers data exploration, preprocessing, feature selection, Machine Learning model training, hyperparameter tuning, model evaluation, model saving, and a Streamlit-based prediction application.

Objective

The objective of this project is to build a Machine Learning model that predicts whether a customer is likely to churn based on selected customer attributes.

Prediction output:

YES → Customer is predicted to churn

NO → Customer is predicted not to churn

Dataset

The dataset was downloaded from Kaggle and is included in the project as:

customer_churn_data.csv

Dataset Information

Rows: 1,000

Columns: 10

Dataset Columns

Column

Description

CustomerID

Unique customer identifier

Age

Customer age

Gender

Customer gender

Tenure

Customer tenure

MonthlyCharges

Monthly customer charges

ContractType

Customer contract type

InternetService

Internet service information

TotalCharges

Total customer charges

TechSupport

Technical support information

Churn

Customer churn status

Exploratory Data Analysis

The Jupyter Notebook includes exploratory analysis using Python and Pandas.

The analysis includes:

Dataset structure inspection

Statistical summary

Missing-value analysis

Duplicate-value checking

Group-based analysis

Churn analysis

Age analysis

Monthly charges analysis

Contract-based analysis

Data visualization

Data Preprocessing

Missing Values

The InternetService column contained missing values.

Missing values were replaced with an empty string:

df['InternetService'] = df['InternetService'].fillna("")

After preprocessing, there were no remaining missing values.

Duplicate Records

Duplicate records were checked using:

df.duplicated().sum()

The result was 0, meaning no duplicate records were found.

Feature Selection

The Machine Learning model uses these four input features:

Age
Gender
Tenure
MonthlyCharges

Target variable:

Churn

Feature Encoding

Gender

Gender was converted into numerical values:

X['Gender'] = X['Gender'].apply(
    lambda x: 1 if x == 'Female' else 0
)

Encoding:

Female → 1
Male   → 0

Churn

The target variable was converted into numerical values:

y['Churn'] = y['Churn'].apply(
    lambda x: 1 if x == 'Yes' else 0
)

Encoding:

Yes → 1
No  → 0

Train-Test Split

The dataset was divided into training and testing data using train_test_split.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)

Split:

80% → Training Data
20% → Testing Data

Feature Scaling

StandardScaler from Scikit-learn was used for feature scaling.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

The scaler was saved using Joblib:

joblib.dump(scaler, 'scaler.pkl')

Machine Learning Models

The following Machine Learning algorithms were tested:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Classifier (SVC)

Decision Tree Classifier

Random Forest Classifier

Hyperparameter Tuning

GridSearchCV was used for hyperparameter tuning.

K-Nearest Neighbors

Parameters searched:

n_neighbors: [3, 5, 7, 9]
weights: ['uniform', 'distance']

Best parameters:

n_neighbors = 7
weights = distance

Support Vector Classifier

Parameters searched:

C: [0.1, 0.1, 0.5, 1]
kernel: ['linear', 'rbf', 'poly']

Best parameters:

C = 0.1
kernel = linear

Decision Tree

Parameters searched included:

criterion
splitter
max_depth
min_samples_split
min_samples_leaf

Best parameters:

criterion = gini
max_depth = None
min_samples_leaf = 4
min_samples_split = 2
splitter = random

Random Forest

Parameters searched included:

n_estimators
max_features
bootstrap

Best parameters:

bootstrap = True
max_features = 2
n_estimators = 64

Model Evaluation

The models were evaluated using Accuracy Score.

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)

The notebook reports approximately 91% accuracy for the evaluated models.

Final Model

The final saved model is the best estimator obtained from the Support Vector Classifier (SVC) GridSearchCV process.

Best SVC parameters:

C = 0.1
kernel = linear

The trained model was saved as:

model.pkl

The feature scaler was saved as:

scaler.pkl

Streamlit Prediction Application

The project includes a Streamlit application in:

app.py

The application loads the saved model and scaler:

scalar = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

Application Inputs

The application accepts:

Age

Gender

Tenure

Monthly Charge

The selected gender is converted into numerical form, the input is transformed using the saved scaler, and the trained model generates the churn prediction.

Prediction Output

The application displays:

Predicted: YES

or:

Predicted: NO

Project Structure

Customer_Churn_Prediction_Using_ML/
│
├── app.py
├── Customer_churn_Analysis.ipynb
├── customer_churn_data.csv
├── model.pkl
├── scaler.pkl
├── README.md
└── .gitignore

Technologies Used

Programming Language

Python

Data Analysis

Pandas

NumPy

Machine Learning

Scikit-learn

Logistic Regression

K-Nearest Neighbors

Support Vector Classifier

Decision Tree

Random Forest

GridSearchCV

StandardScaler

Model Saving

Joblib

Application

Streamlit

Development Tools

Jupyter Notebook

VS Code

Git

GitHub

Dataset

Kaggle

Installation

Clone the repository:

git clone https://github.com/Rajeshkumar0707/customer-churn-prediction-using-ml.git

Navigate to the project directory:

cd customer-churn-prediction-using-ml

Install the required packages:

pip install pandas numpy scikit-learn joblib streamlit

Run the Streamlit Application

Run:

streamlit run app.py

The Streamlit application will open in the browser.

Enter the required customer information and click the Predict! button to generate the churn prediction.

Run the Jupyter Notebook

Open:

Customer_churn_Analysis.ipynb

The notebook contains the Machine Learning workflow:

Data Loading
     ↓
Data Exploration
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Duplicate Checking
     ↓
Feature Selection
     ↓
Feature Encoding
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Model Saving

Git and GitHub

The project was version-controlled using Git and uploaded to GitHub.

Git workflow:

git init
git add .
git status
git commit -m "Initial commit"
git remote add origin https://github.com/Rajeshkumar0707/customer-churn-prediction-using-ml.git
git branch -M main
git push -u origin main

.gitignore

The project uses .gitignore to exclude local and unnecessary files:

venv/
.env
__pycache__/
*.pyc
.ipynb_checkpoints/

GitHub Repository

https://github.com/Rajeshkumar0707/customer-churn-prediction-using-ml

Key Project Highlights

Built a Customer Churn Prediction project using Python and Machine Learning.

Analyzed a 1,000-row customer dataset.

Performed missing-value handling and duplicate checking.

Selected relevant features for churn prediction.

Applied categorical encoding.

Applied feature scaling using StandardScaler.

Tested multiple Machine Learning algorithms.

Used GridSearchCV for hyperparameter tuning.

Evaluated models using Accuracy Score.

Achieved approximately 91% accuracy in the notebook evaluation.

Saved the trained model using Joblib.

Built a Streamlit-based prediction application.

Uploaded the project to GitHub.

Author

Rajesh Kumar

Computer Science Engineering Graduate
Aspiring Data Scientist | Machine Learning Engineer

GitHub: https://github.com/Rajeshkumar0707