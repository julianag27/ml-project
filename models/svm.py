from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd


def svm_run_models(X_train, Y_train, X_test, Y_test):
    # Initializing final stats table
    stats = pd.DataFrame({
        "label": [],
        "desc": [],
        "model_type": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "confusion_matrix": [],
        "ROC/AUC": []
    })

    models = svm_get_models()
    for label, desc, model in models:
        # Generate statistics
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        cm = confusion_matrix(Y_test, Y_pred)

        stats = pd.concat([stats, pd.DataFrame({
            "label": [label],
            "desc": [desc],
            "model_type": ["SVM"],
            "train_accuracy": [model.score(X_train, Y_train)],
            "test_accuracy": [accuracy_score(Y_test, Y_pred)],
            "precision": [precision_score(Y_test, Y_pred)],
            "recall": [recall_score(Y_test, Y_pred)],
            "f1_score": [f1_score(Y_test, Y_pred)],
            "confusion_matrix": [[[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]]], # turns CM from ndarray into a python nested list
            "ROC/AUC": [roc_auc_score(Y_test, Y_pred)]
        })], ignore_index=True)
    return stats

def svm_get_models():
    # add models as functions and add the function to the list below
    return [
        rbf(),
        rbf2(),
        rbf3(),
        rbf4(),
        rbf5(),
        lin(),
        lin2(),
        lin3(),
        poly2(),
        poly3(),
        poly4(),
        poly5(),
        sig(),
        sig2()
    ]

#RBF Models
def rbf():
    return ("RBF C=1 gamma=scale", 
            "\tRegularization Param: 1\n\tKernel: RBF\n\tGamma: scale", 
            SVC(C=1, kernel='rbf', gamma='scale'))

def rbf2():
    return ("RBF C=10 gamma=scale",
            "\tRegularization Param: 10\n\tKernel: RBF\n\tGamma: scale",
            SVC(C=10, kernel='rbf', gamma='scale'))

def rbf3():
    return ("RBF C=100 gamma=scale",
            "\tRegularization Param: 100\n\tKernel: RBF\n\tGamma: scale",
            SVC(C=100, kernel='rbf', gamma='scale'))

def rbf4():
    return ("RBF C=1 gamma=0.05",
            "\tRegularization Param: 1\n\tKernel: RBF\n\tGamma: 0.05",
            SVC(C=1, kernel='rbf', gamma=0.05))

def rbf5():
    return ("RBF C=0.1 gamma=scale",
            "\tRegularization Param: 0.1\n\tKernel: RBF\n\tGamma: scale",
            SVC(C=0.1, kernel='rbf', gamma='scale'))

#Linear Models
def lin():
    return ("Linear C=1",
            "\tRegularization Param: 1\n\tKernel: Linear",
            SVC(C=1, kernel='linear'))

def lin2():
    return ("Linear C=0.1",
            "\tRegularization Param: 0.1\n\tKernel: Linear",
            SVC(C=1, kernel='linear'))

def lin3():
    return ("Linear C=10",
            "\tRegularization Param: 10\n\tKernel: Linear",
            SVC(C=1, kernel='linear'))

#Polynomial Models
def poly2():
    return ("Poly C=1 degree=2",
            "\tRegularization Param: 1\n\tKernel: Polynomial\n\tDegree: 2",
            SVC(C=1, kernel='poly', degree=2))

def poly3():
    return ("Poly C=1 degree=3",
            "\tRegularization Param: 1\n\tKernel: Polynomial\n\tDegree: 3",
            SVC(C=1, kernel='poly', degree=3))

def poly4():
    return ("Poly C=1 degree=4",
            "\tRegularization Param: 1\n\tKernel: Polynomial\n\tDegree: 4",
            SVC(C=1, kernel='poly', degree=4))

def poly5():
    return ("Poly C=1 degree=5",
            "\tRegularization Param: 1\n\tKernel: Polynomial\n\tDegree: 5",
            SVC(C=1, kernel='poly', degree=5))

#Sigmoid Models
def sig():
    return ("Sigmoid C=1 gamma=scale",
            "\tRegularization Param: 1\n\tKernel: Sigmoid\n\tGamma: scale",
            SVC(C=1, kernel='sigmoid', gamma='scale'))

def sig2():
    return ("Sigmoid C=10 gamma=scale",
            "\tRegularization Param: 10\n\tKernel: Sigmoid\n\tGamma: scale",
            SVC(C=10, kernel='sigmoid', gamma='scale'))