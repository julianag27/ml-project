from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def rg_run_models(X_train, Y_train, X_test, Y_test):
    # Initializing final stats table
    stats = pd.DataFrame({
        "label": [],
        "desc": [],
        "model_type": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "confusion_matrix": []
    })

    models = rg_get_models()
    for label, desc, model in models:
        # Generate statistics
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        if isinstance(model, LinearRegression):
            Y_pred = (Y_pred >= 0.5).astype(int)
        
        cm = confusion_matrix(Y_test, Y_pred)

        stats = pd.concat([stats, pd.DataFrame({
            "label": [label],
            "desc": [desc],
            "model_type": ["Regression"],
            "accuracy": [accuracy_score(Y_test, Y_pred)],
            "precision": [precision_score(Y_test, Y_pred)],
            "recall": [recall_score(Y_test, Y_pred)],
            "f1_score": [f1_score(Y_test, Y_pred)],
            "confusion_matrix": [[[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]]] # turns CM from ndarray into a python nested list
        })], ignore_index=True)
    return stats

def rg_get_models():
    # add models as functions and add the function to the list below
    return [
        linear(),
        logistic(),
        logistic_cv()
    ]

### Linear Regression
def linear():
    return ("Linear Regression", "\tDefault Settings", LinearRegression())

### Logistic Regression
def logistic():
    return ("Logistic Regression", "\tClass Weight: Balanced\n\tRandom State: 42", LogisticRegression(class_weight='balanced', random_state=42))

### Logistic Regression CV
def logistic_cv():
    return ("Logistic Regression CV", "\tClass Weight: Balanced\n\tSolver: Saga\n\tRandom State: 42\n\tMax Iterations: 1000", LogisticRegressionCV(class_weight='balanced', solver='saga', random_state=42, max_iter=1000))