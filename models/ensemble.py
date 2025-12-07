from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, StackingClassifier

def ens_run_models(X_train, Y_train, X_test, Y_test):
    """
    Runs all ensemble models defined in ens_get_models and returns a stats DataFrame.
    """
    stats = pd.DataFrame({
        "label": [], "desc": [], "model_type": [], "train_accuracy": [],
        "test_accuracy": [], "precision": [], "recall": [], "f1_score": [],
        "confusion_matrix": [], "ROC/AUC": []
    })

    models = ens_get_models()
    for label, desc, model in models:
        model.fit(X_train, Y_train)
        
        #makes model predict yes more often, made SMOTE models able to get slightly above accuracy that non-SMOTE models got by guessing only yes
        y_prob = model.predict_proba(X_test)[:, 1]
        Y_pred = (y_prob >= 0.4).astype(int)
        cm = confusion_matrix(Y_test, Y_pred)

        stats = pd.concat([stats, pd.DataFrame({
            "label": [label],
            "desc": [desc],
            "model_type": ["ENS"],
            "train_accuracy": [model.score(X_train, Y_train)],
            "test_accuracy": [accuracy_score(Y_test, Y_pred)],
            "precision": [precision_score(Y_test, Y_pred)],
            "recall": [recall_score(Y_test, Y_pred)],
            "f1_score": [f1_score(Y_test, Y_pred)],
            "confusion_matrix": [[[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]]],
            "ROC/AUC": [roc_auc_score(Y_test, Y_pred)]
        })], ignore_index=True)
    return stats

def ens_get_models():
    return [
        rfc(),
        bagging_dtc(),
        adaboost_dtc(),
        stacking_example()
    ]

def rfc():
    return ("Random Forest",
            "Random Forest Classifier with 500 trees, min_samples_split=5, min_samples_leaf=4",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=4,
                max_features='sqrt',
                ))

def bagging_dtc():
    return ("Bagging Decision Tree",
            "Bagging Classifier with 100 Decision Tree base estimators",
            BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=None, min_samples_split=5),
                n_estimators=100,
            ))

def adaboost_dtc():
    return ("AdaBoost Decision Tree",
            "AdaBoost with 250 shallow Decision Trees as weak learners",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=4),
                n_estimators=250,
                learning_rate=0.3,
                
            ))

def stacking_example():
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200)),
        ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(max_depth=5))
    ]
    return ("Stacking RF+GB+KNN+DT",
            "Stacking Classifier combining RF, GB, KNN, DT with Logistic Regression final estimator",
            StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=2000),
                cv=5
            ))
