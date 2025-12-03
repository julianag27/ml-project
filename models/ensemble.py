from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier




def ens_run_models(X_train, Y_train, X_test, Y_test):
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

    models = ens_get_models()
    for label, desc, model in models:
        # Generate statistics
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        cm = confusion_matrix(Y_test, Y_pred)

        stats = pd.concat([stats, pd.DataFrame({
            "label": [label],
            "desc": [desc],
            "model_type": ["ENS"],
            "accuracy": [accuracy_score(Y_test, Y_pred)],
            "precision": [precision_score(Y_test, Y_pred)],
            "recall": [recall_score(Y_test, Y_pred)],
            "f1_score": [f1_score(Y_test, Y_pred)],
            "confusion_matrix": [[[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]]] # turns CM from ndarray into a python nested list
        })], ignore_index=True)
    return stats

def ens_get_models():
    # add models as functions and add the function to the list below
    return [
        rfc()
    ]

def rfc():
    return ("Random Forest 100 estimators max depth 10", "\tRandom Forest Classifier\n\tN Estimators: 100\n\tMax Depth: 10", RandomForestClassifier(n_estimators=100, max_depth=10))