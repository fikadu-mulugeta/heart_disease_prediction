from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_decision_tree(X_train, X_val, y_train, y_val):
    model = DecisionTreeClassifier(min_samples_split=50, max_depth=4, random_state=55)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(model.predict(X_train), y_train)
    val_acc = accuracy_score(model.predict(X_val), y_val)
    print(f"Decision Tree: Train Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")