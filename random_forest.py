from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_random_forest(X_train, X_val, y_train, y_val):
    model = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_split=10, random_state=55)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(model.predict(X_train), y_train)
    val_acc = accuracy_score(model.predict(X_val), y_val)
    print(f"Random Forest: Train Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")