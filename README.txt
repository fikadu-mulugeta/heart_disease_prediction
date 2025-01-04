# Heart Disease Prediction Project

## Overview
This project uses machine learning models to predict the likelihood of heart disease based on patient data. 
It implements Decision Tree, Random Forest, and XGBoost models using the Kaggle Heart Failure Prediction Dataset.

This project was inspired by (and expanded upon) the practical lab of the online course "Advanced Learning Algorithms". 
The course is part of the Machine Learning Specialization, a foundational online program created in collaboration between DeepLearning.AI and Stanford Online, and taught by Andrew Ng, an AI visionary who has led critical research at Stanford University and other prominent organizations.
## Project Structure
```
heart_disease_prediction/
|- main.py
|- decision_tree.py
|- random_forest.py
|- xgboost_model.py
|- results.txt
|- License.txt
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
It is obtained from Kaggle using the following link: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction  

## Results
Decision Tree: Train Accuracy = 86.65%, Validation Accuracy = 86.96%
Random Forest: Train Accuracy = 93.05%, Validation Accuracy = 89.13%
XGBoost: Train Accuracy = 96.73%, Validation Accuracy = 88.59%

## License
This project is licensed under the MIT License. See the License.txt file for details.
