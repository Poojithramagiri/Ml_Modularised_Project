from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 

def train_models(X, y):
    """Trains and evaluates classification models with cross-validation.

    Args:
        X (array-like): Input features.
        y (array-like): Target variable.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)) 
    ]

    for name, model in models:
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # Cross-Validation 
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy') 
        print(f"Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})") 
