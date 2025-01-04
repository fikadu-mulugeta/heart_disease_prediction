# --- README.md ---
# Heart Disease Prediction Project

## Overview
This project uses machine learning models to predict the likelihood of heart disease based on patient data. 
It implements Decision Tree, Random Forest, and XGBoost models using the Kaggle Heart Failure Prediction Dataset.

## Project Structure
```
heart_disease_prediction/
|- main.py
|- decision_tree.py
|- random_forest.py
|- xgboost_model.py
|- results.txt
|- README.md
```

## How to Run
1. Ensure you have the required Python libraries installed:
   ```
   pip install -r requirements.txt
   ```
2. Place the dataset (`heart.csv`) in the project directory.
3. Run the main program:
   ```
   python main.py
   ```

## Dataset
The dataset contains features such as age, gender, cholesterol levels, and more to predict heart disease.

## Results
Decision Tree: Train Accuracy = 86.65%, Validation Accuracy = 86.96%
Random Forest: Train Accuracy = 93.05%, Validation Accuracy = 89.13%
XGBoost: Train Accuracy = 96.73%, Validation Accuracy = 88.59%