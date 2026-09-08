# 📉 Customer Churn Prediction Using Machine Learning

## 📌 Project Summary

This project focuses on predicting whether a customer is likely to churn using Machine Learning.

The project analyzes customer information such as age, gender, tenure, and monthly charges to predict customer churn.

The complete workflow includes:

**Data Collection → Data Understanding → Data Cleaning → Exploratory Data Analysis → Feature Selection → Feature Encoding → Train/Test Split → Feature Scaling → Model Training → Hyperparameter Tuning → Model Evaluation → Model Saving → Streamlit Application → Churn Prediction**

---

## 🎯 Business Problem

Customer churn is an important business problem because losing existing customers can negatively affect revenue and business growth.

Businesses can use customer data to identify customers who are more likely to churn and take preventive actions such as:

- Improving customer support
- Offering personalized plans
- Providing suitable discounts
- Improving service quality
- Identifying high-risk customers
- Developing customer retention strategies

### ❓ Business Question

> Can customer information be used to predict whether a customer is likely to churn?

---

## 🎯 Project Objectives

- Analyze customer churn data.
- Understand customer characteristics.
- Perform data cleaning and preprocessing.
- Perform Exploratory Data Analysis (EDA).
- Identify useful features for churn prediction.
- Convert categorical variables into numerical values.
- Split the data into training and testing sets.
- Apply feature scaling.
- Train multiple Machine Learning classification models.
- Perform hyperparameter tuning using GridSearchCV.
- Compare model performance.
- Select a suitable final model.
- Save the trained model using Joblib.
- Save the scaler used for preprocessing.
- Build an interactive Streamlit application.
- Predict customer churn using user-provided customer information.

---

# 🔄 Project Workflow

```text
📊 Customer Churn Dataset
          ↓
🔍 Data Understanding
          ↓
🧹 Data Cleaning
          ↓
📈 Exploratory Data Analysis
          ↓
🎯 Feature Selection
          ↓
🔢 Feature Encoding
          ↓
✂️ Train / Test Split
          ↓
📏 Feature Scaling
          ↓
🤖 Model Training
          ↓
⚙️ Hyperparameter Tuning
          ↓
📊 Model Evaluation
          ↓
🏆 Final Model Selection
          ↓
💾 Save Model + Scaler
          ↓
🌐 Streamlit Application
          ↓
🔮 Customer Churn Prediction
```

---

# 📂 Dataset

The dataset used in this project was downloaded from **Kaggle**.

The dataset contains:

- **1,000 rows**
- **10 columns**

## Dataset Columns

| Column | Description |
|---|---|
| `CustomerID` | Unique identifier for each customer |
| `Age` | Age of the customer |
| `Gender` | Gender of the customer |
| `Tenure` | Duration of the customer's relationship with the service |
| `MonthlyCharges` | Monthly amount charged to the customer |
| `ContractType` | Type of contract associated with the customer |
| `InternetService` | Internet service information |
| `TotalCharges` | Total charges associated with the customer |
| `TechSupport` | Technical support information |
| `Churn` | Indicates whether the customer churned |

### 🎯 Target Variable

The target variable is:

```text
Churn
```

Target values are encoded as:

```text
Yes → 1
No  → 0
```

---

# 🔍 Data Understanding

The dataset was loaded using Pandas.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("customer_churn_data.csv")
```

The dataset was inspected using:

```python
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

The dataset contains:

```text
Rows    : 1,000
Columns : 10
```

---

# 🧹 Data Cleaning

The dataset was checked for:

- Missing values
- Duplicate records
- Data types
- Data distribution
- Categorical values

## Missing Values

The `InternetService` column contained missing values.

The missing values were handled using:

```python
df['InternetService'] = df['InternetService'].fillna("")
```

## Duplicate Records

Duplicate records were checked using:

```python
df.duplicated().sum()
```

Result:

```text
0 duplicate records
```

---

# 📊 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand customer characteristics and identify patterns related to churn.

The analysis included:

- Customer age distribution
- Customer churn distribution
- Gender distribution
- Tenure analysis
- Monthly charges analysis
- Contract type analysis
- Customer characteristics by churn status

Example:

```python
df.groupby('Churn')['Age'].mean()
```

This analysis helps compare the average age of customers based on their churn status.

Another example:

```python
df.groupby('ContractType')['MonthlyCharges'].mean()
```

This helps analyze average monthly charges across different contract types.

Charts were also created during the notebook to visualize the data and identify customer behavior patterns.

---

# 🎯 Feature Selection

For the final Machine Learning model, the following four features were selected:

```text
Age
Gender
Tenure
MonthlyCharges
```

The target variable is:

```text
Churn
```

Feature selection:

```python
X = df[['Age', 'Gender', 'Tenure', 'MonthlyCharges']]
```

Target:

```python
y = df[['Churn']]
```

---

# 🔢 Feature Encoding

Machine Learning models require numerical values.

Therefore, categorical variables were converted into numerical values.

## Gender Encoding

The `Gender` column was encoded as:

```python
X['Gender'] = X['Gender'].apply(
    lambda x: 1 if x == 'Female' else 0
)
```

Encoding:

```text
Female → 1
Male   → 0
```

## Churn Encoding

The target variable was encoded as:

```python
y['Churn'] = y['Churn'].apply(
    lambda x: 1 if x == 'Yes' else 0
)
```

Encoding:

```text
Yes → 1
No  → 0
```

---

# ✂️ Train/Test Split

The dataset was divided into training and testing data using `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)
```

The split was:

```text
80% → Training Data
20% → Testing Data
```

The training data was used for model training and the testing data was used for model evaluation.

> **Note:** `random_state` was not specified in the notebook, so model results may vary between different runs.

---

# 📏 Feature Scaling

`StandardScaler` was used for feature scaling.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
```

The scaler was saved using Joblib:

```python
import joblib

joblib.dump(scaler, 'scaler.pkl')
```

The test data was transformed using the fitted scaler:

```python
X_test = scaler.transform(X_test)
```

---

# 🤖 Machine Learning Models

The following classification algorithms were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Classifier (SVC)
4. Decision Tree Classifier
5. Random Forest Classifier

---

# 1️⃣ Logistic Regression

Logistic Regression was used as a classification model for predicting customer churn.

The observed accuracy was approximately:

```text
91%
```

---

# 2️⃣ K-Nearest Neighbors (KNN)

K-Nearest Neighbors was evaluated and hyperparameters were tuned using `GridSearchCV`.

The parameter grid included:

```python
{
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}
```

Best parameters found:

```text
n_neighbors = 7
weights     = distance
```

The tuned KNN model was evaluated as part of the model comparison.

---

# 3️⃣ Support Vector Classifier (SVC)

Support Vector Classifier was trained for the churn classification problem.

Hyperparameter tuning was performed using `GridSearchCV`.

The parameter grid included:

```python
{
    'C': [0.1, 0.1, 0.5, 1],
    'kernel': ['linear', 'rbf', 'poly']
}
```

Best parameters found:

```text
C      = 0.1
kernel = linear
```

Observed accuracy:

```text
91%
```

---

# 4️⃣ Decision Tree Classifier

A Decision Tree Classifier was also evaluated.

The following parameters were tuned:

```python
{
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

Best parameters found:

```text
criterion         = gini
max_depth         = None
min_samples_leaf  = 4
min_samples_split = 2
splitter          = random
```

Observed accuracy:

```text
91%
```

---

# 5️⃣ Random Forest Classifier

Random Forest Classifier was evaluated using hyperparameter tuning.

The parameter grid included:

```python
{
    'n_estimators': [32, 64, 128, 256],
    'max_features': [2, 3, 4],
    'bootstrap': [True, False]
}
```

Best parameters found:

```text
bootstrap     = True
max_features = 2
n_estimators = 64
```

Observed accuracy:

```text
91%
```

---

# ⚙️ Hyperparameter Tuning

`GridSearchCV` was used to search through different hyperparameter combinations.

The general process was:

```text
🤖 Machine Learning Model
          ↓
⚙️ Define Hyperparameter Grid
          ↓
🔎 GridSearchCV
          ↓
🧪 Test Parameter Combinations
          ↓
🏆 Select Best Parameters
          ↓
📊 Evaluate Best Estimator
```

This helps identify a suitable configuration for each Machine Learning model.

---

# 📊 Model Evaluation

The main evaluation metric used in this project was:

```text
Accuracy
```

Accuracy was calculated using:

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, predictions)
```

Observed model performance:

| Model | Observed Accuracy |
|---|---:|
| Logistic Regression | 91% |
| KNN | ~91% |
| Support Vector Classifier | 91% |
| Decision Tree | 91% |
| Random Forest | 91% |

> **Note:** Accuracy can vary between runs because the notebook uses `train_test_split()` without a fixed `random_state`.

---

# 🏆 Final Model

The final model saved for the Streamlit application is the tuned **Support Vector Classifier (SVC)**.

Best SVC parameters:

```text
C      = 0.1
kernel = linear
```

The best estimator was saved using:

```python
best_model = gridsvc.best_estimator_

joblib.dump(best_model, 'model.pkl')
```

Therefore:

```text
model.pkl  → Trained SVC model
scaler.pkl → Saved StandardScaler
```

---

# 💾 Model Serialization

Joblib was used to save the trained Machine Learning model and scaler.

## Save Model

```python
joblib.dump(best_model, 'model.pkl')
```

## Save Scaler

```python
joblib.dump(scaler, 'scaler.pkl')
```

These files allow the Streamlit application to load the saved model and scaler without retraining the model every time.

---

# 🌐 Streamlit Application

A web application was created using **Streamlit**.

The application allows users to enter customer information and receive a churn prediction.

## User Inputs

The application accepts:

```text
👤 Age
👤 Gender
📅 Tenure
💰 Monthly Charge
```

The user enters the required values and clicks:

```text
🔮 Predict!
```

The application then displays the predicted churn result.

---

# 🔮 Prediction Flow

```text
👤 User Enters Customer Information
              ↓
🔢 Convert Gender to Numerical Value
              ↓
📦 Create Feature Array
              ↓
📏 Transform Input Using Saved Scaler
              ↓
🤖 Load Saved SVC Model
              ↓
🔮 Generate Prediction
              ↓
📊 Display YES / NO
```

The application loads the saved model and scaler:

```python
scalar = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
```

Gender is converted as:

```text
Female → 1
Male   → 0
```

The four input features are:

```text
Age
Gender
Tenure
MonthlyCharges
```

The input is transformed using the saved scaler before being passed to the saved Machine Learning model.

---

# 🖥️ Streamlit Prediction Output

After entering the customer information and clicking the prediction button, the application displays one of the following results:

```text
Predicted: YES
```

or:

```text
Predicted: NO
```

The application also displays a balloon animation after a prediction.

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Data Analysis

- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn

## Machine Learning Algorithms

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier
- Decision Tree Classifier
- Random Forest Classifier

## Model Optimization

- GridSearchCV

## Preprocessing

- StandardScaler
- Manual categorical encoding

## Model Serialization

- Joblib

## Web Application

- Streamlit

## Development Tools

- Jupyter Notebook
- VS Code

## Version Control

- Git
- GitHub

---

# 📁 Project Structure

```text
Customer_Churn_Prediction_Using_ML/
│
├── 📄 app.py
├── 📓 Customer_churn_Analysis.ipynb
├── 📊 customer_churn_data.csv
├── 🤖 model.pkl
├── 📏 scaler.pkl
└── 📄 .gitignore
```

---

# 📄 File Description

| File | Description |
|---|---|
| `app.py` | Streamlit application used for customer churn prediction |
| `Customer_churn_Analysis.ipynb` | Notebook containing data analysis, preprocessing, model training and evaluation |
| `customer_churn_data.csv` | Customer churn dataset |
| `model.pkl` | Saved tuned SVC Machine Learning model |
| `scaler.pkl` | Saved StandardScaler |
| `.gitignore` | Files and folders excluded from Git |

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rajeshkumar0707/customer-churn-prediction-using-ml.git
```

## 2️⃣ Navigate to the Project Directory

```bash
cd customer-churn-prediction-using-ml
```

## 3️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

## 4️⃣ Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 5️⃣ Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

---

# ▶️ Run the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Customer_churn_Analysis.ipynb
```

Run the notebook cells sequentially to reproduce the data analysis, preprocessing, model training and evaluation workflow.

---

# ▶️ Run the Streamlit Application

From the project directory, run:

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

# 🚀 Complete End-to-End Workflow

```text
📥 Dataset
   ↓
📊 Data Loading
   ↓
🔍 Data Understanding
   ↓
🧹 Data Cleaning
   ↓
📈 Exploratory Data Analysis
   ↓
🎯 Feature Selection
   ↓
🔢 Feature Encoding
   ↓
✂️ Train/Test Split
   ↓
📏 Feature Scaling
   ↓
🤖 Model Training
   ↓
⚙️ GridSearchCV
   ↓
📊 Model Evaluation
   ↓
🏆 Final SVC Selection
   ↓
💾 model.pkl
   ↓
💾 scaler.pkl
   ↓
🌐 Streamlit App
   ↓
🔮 Customer Churn Prediction
```

---

# 📌 Key Highlights

- Built an end-to-end Machine Learning classification project.
- Used a customer churn dataset containing 1,000 records.
- Performed data cleaning and preprocessing.
- Handled missing values in the dataset.
- Checked for duplicate records.
- Performed Exploratory Data Analysis.
- Selected four features for the final prediction model.
- Encoded categorical variables.
- Used an 80/20 train-test split.
- Applied StandardScaler.
- Trained five Machine Learning classification models.
- Used GridSearchCV for hyperparameter tuning.
- Compared multiple classification algorithms.
- Selected a tuned SVC as the final saved model.
- Saved the trained model using Joblib.
- Saved the preprocessing scaler using Joblib.
- Built an interactive Streamlit application.
- Implemented customer churn prediction from user inputs.

---

# 📚 Skills Demonstrated

## 🐍 Python

- Python programming
- Pandas
- NumPy
- Data manipulation
- Data preprocessing

## 📊 Data Analysis

- Exploratory Data Analysis
- Descriptive statistics
- GroupBy analysis
- Missing value handling
- Duplicate checking
- Data visualization

## 🤖 Machine Learning

- Supervised Learning
- Binary Classification
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier
- Decision Tree
- Random Forest
- Hyperparameter tuning
- GridSearchCV
- Model evaluation

## 📏 Data Preprocessing

- Feature selection
- Feature encoding
- Standardization
- Train/Test splitting

## 🌐 Application Development

- Streamlit
- Joblib model loading
- Interactive prediction

## 🔧 Development Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

# 📈 Project Outcome

This project demonstrates how customer information can be analyzed and used to build a Machine Learning classification model for predicting customer churn.

Multiple classification models were trained and evaluated, with observed accuracy around **91%** for the evaluated models.

The tuned **Support Vector Classifier (SVC)** was selected as the final saved model.

The trained model was saved as:

```text
model.pkl
```

The scaler was saved as:

```text
scaler.pkl
```

A Streamlit application was then developed to allow users to enter customer information and receive a churn prediction.

---

# 🔮 Future Improvements

The project can be improved further by:

- Using a Scikit-learn `Pipeline` to ensure preprocessing and model training use the same transformation workflow.
- Including additional relevant features such as `ContractType`, `InternetService`, `TotalCharges`, and `TechSupport`.
- Using a fixed `random_state` for reproducible results.
- Using stratified train/test splitting when appropriate.
- Evaluating Precision, Recall, F1-score and ROC-AUC.
- Adding a confusion matrix.
- Performing cross-validation.
- Checking and handling class imbalance if required.
- Performing additional feature engineering.
- Adding model explainability.
- Adding prediction probability to the Streamlit application.
- Improving the Streamlit user interface.
- Deploying the application to a cloud platform.

---

# 🔐 Security & Best Practices

The project uses a `.gitignore` file to exclude unnecessary files and local environment files from Git.

The `.gitignore` contains:

```text
venv/
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
```

Best practices include:

- Keeping the virtual environment out of Git.
- Avoiding unnecessary temporary files.
- Keeping model files separate from source code.
- Organizing the project into clear and understandable files.
- Avoiding sensitive information in source code.

---

# 🔗 GitHub Repository

👉 **[Customer Churn Prediction Using Machine Learning](https://github.com/Rajeshkumar0707/customer-churn-prediction-using-ml)**

---

# 👨‍💻 Author

## Rajesh Kumar

**Computer Science Engineering Graduate | Aspiring Data Scientist | Machine Learning Enthusiast**

### GitHub

👉 [Rajesh Kumar GitHub](https://github.com/Rajeshkumar0707)

---

# ⭐ Project Summary

```text
📊 Customer Churn Dataset
        ↓
🐍 Python + Pandas + NumPy
        ↓
📈 EDA + Data Preprocessing
        ↓
🤖 Scikit-learn
        ↓
⚙️ Multiple ML Models
        ↓
🔎 GridSearchCV
        ↓
🏆 Tuned SVC
        ↓
💾 Joblib
        ↓
🌐 Streamlit
        ↓
🔮 Customer Churn Prediction
```

⭐ If you find this project useful, consider giving the repository a star!

---

## 🚀 Thank You

Thank you for visiting this project.

This project demonstrates a complete Machine Learning workflow, from **data analysis and preprocessing to model training, hyperparameter tuning, evaluation, model serialization, and interactive Streamlit prediction**.
