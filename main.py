import pandas as pd
from sklearn.model_selection import train_test_split
from decision_tree import train_decision_tree
from random_forest import train_random_forest
from xgboost_model import train_xgboost

RANDOM_STATE = 55

# Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    cat_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df = pd.get_dummies(data=df, prefix=cat_variables, columns=cat_variables)
    features = [x for x in df.columns if x != 'HeartDisease']
    X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size=0.8, random_state=RANDOM_STATE)
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    filepath = "heart.csv"  # Update this path as needed
    X_train, X_val, y_train, y_val = load_and_preprocess_data(filepath)

    print("Training Decision Tree...")
    train_decision_tree(X_train, X_val, y_train, y_val)

    print("Training Random Forest...")
    train_random_forest(X_train, X_val, y_train, y_val)

    print("Training XGBoost...")
    train_xgboost(X_train, X_val, y_train, y_val)
RANDOM_STATE = 55

# Load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    cat_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df = pd.get_dummies(data=df, prefix=cat_variables, columns=cat_variables)
    features = [x for x in df.columns if x != 'HeartDisease']
    X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size=0.8, random_state=RANDOM_STATE)
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    filepath = "heart.csv"  # Update this path as needed
    X_train, X_val, y_train, y_val = load_and_preprocess_data(filepath)

    print("Training Decision Tree...")
    train_decision_tree(X_train, X_val, y_train, y_val)

    print("Training Random Forest...")
    train_random_forest(X_train, X_val, y_train, y_val)

    print("Training XGBoost...")
    train_xgboost(X_train, X_val, y_train, y_val)