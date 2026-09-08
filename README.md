Customer Churn Prediction Using Machine Learning

📌 Project Overview

Customer churn is a major challenge for businesses that provide subscription-based or long-term services. Losing customers can directly affect revenue, customer relationships, and business growth.

This project focuses on building a Machine Learning-based Customer Churn Prediction system that analyzes customer information and predicts whether a customer is likely to churn.

The project follows an end-to-end Machine Learning workflow, starting from dataset collection and exploratory data analysis to data preprocessing, model training, evaluation, model serialization, and prediction through a Flask application.

🎯 Project Objective

The main objective of this project is to develop a machine learning classification system that can identify customers who are likely to leave a service.

The project aims to:

Analyze customer data

Understand customer churn patterns

Perform exploratory data analysis

Clean and preprocess the dataset

Prepare features for machine learning

Train a classification model

Evaluate model performance

Save the trained model

Save the feature scaler

Build a Flask application for predictions

💼 Business Problem

Customer acquisition can be more expensive than retaining existing customers.

If a business can identify customers who are likely to churn, it can take preventive actions such as:

Providing personalized offers

Improving customer support

Offering discounts or incentives

Targeting high-risk customers

Improving customer experience

Developing customer retention strategies

Therefore, a churn prediction model can help businesses make data-driven customer retention decisions.

📊 Dataset

The dataset used in this project was downloaded from Kaggle and used for customer churn analysis and machine learning model development.

The downloaded dataset is included in this project as:

customer_churn_data.csv

Dataset Source

Source: Kaggle

Dataset Type: Customer Churn Dataset

Format: CSV

Note: Add the exact Kaggle dataset URL here if you want to reference the original Kaggle page.

🔄 End-to-End Machine Learning Workflow

Kaggle Dataset
      ↓
Download Dataset
      ↓
Load Dataset
      ↓
Data Understanding
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Selection
      ↓
Data Preprocessing
      ↓
Feature Scaling
      ↓
Train-Test Split
      ↓
Machine Learning Model
      ↓
Model Evaluation
      ↓
Save Trained Model
      ↓
Save Scaler
      ↓
Flask Application
      ↓
Customer Churn Prediction

🔍 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the structure and characteristics of the customer dataset.

The analysis includes:

Dataset dimensions

Column information

Data types

Missing-value analysis

Duplicate-value analysis

Statistical analysis

Numerical feature analysis

Categorical feature analysis

Churn distribution

Relationships between customer attributes and churn

EDA helps identify important patterns and prepares the dataset for machine learning.

🧹 Data Preprocessing

Before training the machine learning model, the dataset was prepared through preprocessing steps.

The preprocessing workflow includes:

Checking missing values

Handling data inconsistencies

Preparing categorical variables

Selecting relevant features

Preparing the target variable

Splitting the dataset into training and testing data

Scaling features where required

The feature scaler used during preprocessing is saved as:

scaler.pkl

This allows new customer data to be transformed using the same scaling process used during model training.

🤖 Machine Learning

This project uses Supervised Machine Learning to solve a customer churn classification problem.

The model learns patterns from historical customer data and uses those patterns to predict the churn outcome for new customers.

Machine Learning Type

Supervised Learning
        ↓
Classification
        ↓
Customer Churn Prediction

The trained machine learning model is saved as:

model.pkl

Saving the trained model allows it to be reused without retraining every time the application starts.

📈 Model Evaluation

The trained classification model is evaluated using appropriate classification metrics.

The evaluation process can include:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

These metrics help determine how effectively the model identifies customers who are likely to churn.

Actual performance values should be added here based on the results from Customer_churn_Analysis.ipynb. No accuracy or other metric is assumed without verification.

🌐 Flask Application

A Flask application is included to use the trained machine learning model for customer churn predictions.

The Flask application is implemented in:

app.py

The application loads:

model.pkl
scaler.pkl

and uses them to generate predictions from customer input.

Prediction Workflow

Customer Input
      ↓
Input Validation
      ↓
Feature Preparation
      ↓
Feature Scaling
      ↓
Load Trained Model
      ↓
Prediction
      ↓
Churn Result

💾 Saved Machine Learning Files

model.pkl

Contains the trained machine learning model.

The saved model can be loaded later for making predictions without retraining the model.

scaler.pkl

Contains the feature scaling object used during preprocessing.

The same scaler is used to transform new input data before passing it to the trained model.

🛠️ Technologies Used

Programming Language

Python

Data Analysis

Pandas

NumPy

Machine Learning

Scikit-learn

Supervised Learning

Classification

Feature Scaling

Model Evaluation

Web Framework

Flask

Development Tools

Jupyter Notebook

VS Code

Git

GitHub

Dataset Source

Kaggle

📂 Project Structure

Customer_Churn_Prediction_Using_ML/
│
├── Customer_churn_Analysis.ipynb
├── customer_churn_data.csv
├── model.pkl
├── scaler.pkl
├── app.py
├── README.md
└── .gitignore

📄 File Description

File

Description

Customer_churn_Analysis.ipynb

Complete data analysis, preprocessing, model training, and evaluation workflow

customer_churn_data.csv

Customer churn dataset downloaded from Kaggle

model.pkl

Serialized trained machine learning model

scaler.pkl

Saved feature scaler used during preprocessing

app.py

Flask application used for customer churn prediction

README.md

Project documentation

.gitignore

Specifies files and folders that should not be uploaded to GitHub

⚙️ Installation and Setup

1. Clone the Repository

git clone https://github.com/Rajeshkumar0707/customer-churn-prediction-using-ml.git

2. Navigate to the Project Directory

cd customer-churn-prediction-using-ml

3. Create a Virtual Environment

python -m venv venv

4. Activate the Virtual Environment

For Windows:

venv\Scripts\activate

📦 Install Required Libraries

Install the required Python packages:

pip install pandas numpy scikit-learn flask

If a requirements.txt file is added later, dependencies can be installed using:

pip install -r requirements.txt

▶️ Run the Jupyter Notebook

Open the project in VS Code or Jupyter Notebook.

Open:

Customer_churn_Analysis.ipynb

Run the notebook cells to perform:

Data loading

Data analysis

Data preprocessing

Model training

Model evaluation

Model saving

▶️ Run the Flask Application

After installing the required dependencies, run:

python app.py

The Flask application will start locally.

Open the local URL displayed in the terminal to access the application.

🧪 Testing

The project can be tested through the following workflow.

Machine Learning Testing

Load the dataset

Run the notebook

Verify data preprocessing

Train the model

Evaluate the model

Save the model

Save the scaler

Flask Application Testing

Start the Flask application

Provide customer information

Submit the input

Verify that the application processes the input

Check the predicted churn result

🔐 GitHub Security

The following files and folders should not be uploaded to GitHub:

venv/
.env
__pycache__/
.ipynb_checkpoints/
*.pyc

These files are excluded through .gitignore.

Never store the following information directly in a public repository:

Passwords

API keys

Database credentials

Secret tokens

Private configuration values

🌱 Git and GitHub Workflow

This project is managed using Git and uploaded to GitHub.

Initialize Git Repository

git init

Add Project Files

git add .

Check Git Status

git status

Create Initial Commit

git commit -m "Initial commit"

Create GitHub Repository

Create a new repository on GitHub with the name:

customer-churn-prediction-using-ml

Recommended settings:

Repository visibility: Public

Add README: No

Add .gitignore: No

Add license: No

The README and .gitignore are already present in the local project.

Connect Local Repository to GitHub

git remote add origin https://github.com/Rajeshkumar0707/customer-churn-prediction-using-ml.git

Rename Branch to Main

git branch -M main

Push Project to GitHub

git push -u origin main

After the push completes, the complete project will be available in the GitHub repository.

🔄 Updating the Project

Whenever changes are made to the project, use:

git add .

git commit -m "Update project"

git push

This keeps the GitHub repository updated with the latest project changes.

💡 Business Applications

Customer churn prediction can be useful in industries such as:

Telecommunications

Banking

Insurance

Subscription services

E-commerce

SaaS

Retail

Financial services

Potential business uses include:

Customer retention

Customer segmentation

Targeted marketing

Personalized offers

Risk identification

Customer relationship management

Reducing customer attrition

🚀 Future Enhancements

Possible improvements for this project include:

Comparing multiple machine learning algorithms

Hyperparameter tuning

Cross-validation

Feature importance analysis

Model explainability

SHAP-based model interpretation

Interactive dashboard

Improved Flask user interface

REST API integration

Cloud deployment

Model monitoring

Automated ML pipeline

Real-time prediction system

📚 Skills Demonstrated

This project demonstrates practical knowledge of:

Python

Pandas

NumPy

Data Cleaning

Exploratory Data Analysis

Data Preprocessing

Feature Engineering

Feature Scaling

Supervised Machine Learning

Classification

Model Evaluation

Model Serialization

Flask

Git

GitHub

Kaggle Dataset Handling

🎓 Learning Outcomes

Through this project, the following practical skills were developed:

Understanding a real-world machine learning problem

Working with a real customer dataset

Performing exploratory data analysis

Preparing data for machine learning

Building a classification model

Evaluating machine learning performance

Saving trained machine learning models

Integrating a trained model with Flask

Managing a project using Git

Publishing a machine learning project on GitHub

📌 Project Highlights

End-to-end Machine Learning project

Real-world customer churn prediction problem

Dataset downloaded from Kaggle

Exploratory Data Analysis

Data preprocessing

Feature scaling

Classification model

Model evaluation

Saved ML model

Saved preprocessing scaler

Flask prediction application

Git and GitHub version control

👨‍💻 Author

Rajesh Kumar

Computer Science Engineering Graduate

Aspiring Data Scientist | Machine Learning Engineer

GitHub:

https://github.com/Rajeshkumar0707

⭐ Repository

Project Repository:

https://github.com/Rajeshkumar0707/customer-churn-prediction-using-ml

If you find this project useful, consider giving the repository a ⭐.