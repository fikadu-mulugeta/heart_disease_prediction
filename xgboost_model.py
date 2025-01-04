from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_xgboost(X_train, X_val, y_train, y_val):
    model = XGBClassifier(n_estimators=500, learning_rate=0.1, random_state=55, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    train_acc = accuracy_score(model.predict(X_train), y_train)
    val_acc = accuracy_score(model.predict(X_val), y_val)
    print(f"XGBoost: Train Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")