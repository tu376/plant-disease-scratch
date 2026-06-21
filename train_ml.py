import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import plot_importance

from model.xgboost import XGBoostModel
from model.svm import SVMModel
from model.random_forest import RandomForestModel
from model.logistic_regression import LogisticRegressionModel
from model.knn import KNeighborsClassifier

X_train = np.load("features/X_train.npy")
y_train = np.load("features/y_train.npy")
X_val = np.load("features/X_val.npy")
y_val = np.load("features/y_val.npy")
X_test = np.load("features/X_test.npy")
y_test = np.load("features/y_test.npy")
classes = np.load("features/classes.npy", allow_pickle=True)

num_classes = len(classes)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["xgboost", "svm", "random_forest", "knn", "logistic_regression"],
    required=True
)
args = parser.parse_args()

if args.model == "xgboost":
    xgb = XGBoostModel(
        num_classes=num_classes,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )
    print("Training XGBoost...")
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y_test, pred, target_names=classes, digits=4))
    
    plot_importance(xgb.model, importance_type='gain', max_num_features=20)
    plt.title("Feature Importance")
    plt.show()

elif args.model == "svm":
    C_values = [0.01, 0.1, 1, 10, 100]
    accuracies, f1_scores = [], []
    
    for c_value in C_values:
        print(f"Training SVM with C={c_value}...")
        svm = SVMModel(C=c_value, kernel="rbf", gamma="scale")
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        
    best_acc_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1_scores)
    
    print(f"Best test Accuracy: {accuracies[best_acc_idx]:.4f} at C={C_values[best_acc_idx]}")
    print(f"Best F1_score: {f1_scores[best_f1_idx]:.4f} at C={C_values[best_f1_idx]}")
    
    plt.figure(figsize=(8,5))
    plt.plot(C_values, accuracies, marker='o')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Accuracy (%)')
    plt.title('SVM Accuracy vs C')
    plt.grid(True)
    plt.show()

elif args.model == "random_forest":
    n_trees = [10, 50, 100, 200, 300]
    accuracies, f1_scores = [], []
    
    for n in n_trees:
        print(f"Training Random Forest with n_estimators={n}...")
        rf = RandomForestModel(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        
    best_acc_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1_scores)
    
    print(f"Best test Accuracy: {accuracies[best_acc_idx]:.4f} at n_estimators={n_trees[best_acc_idx]}")
    print(f"Best F1_score: {f1_scores[best_f1_idx]:.4f} at n_estimators={n_trees[best_f1_idx]}")
    
    plt.plot(n_trees, accuracies, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Accuracy vs Number of Trees')
    plt.grid(True)
    plt.show()

elif args.model == "knn":
    n_neighbors = [1, 3, 5, 7, 9]
    accuracies, f1_scores = [], []
    
    for k in n_neighbors:
        print(f"Training KNN with k={k}...")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        
    best_acc_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1_scores)
        
    print(f"Best test Accuracy: {accuracies[best_acc_idx]:.4f} at k={n_neighbors[best_acc_idx]}")
    print(f"Best F1_score: {f1_scores[best_f1_idx]:.4f} at k={n_neighbors[best_f1_idx]}")
    
    plt.plot(n_neighbors, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy (%)')
    plt.title('KNN Accuracy vs Number of Neighbors')
    plt.grid(True)
    plt.show()

elif args.model == "logistic_regression":
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    accuracies, f1_scores = [], []

    for C in C_values:
        print(f"Training Logistic Regression with C={C}...")
        model = LogisticRegressionModel(C=C, max_iter=5000)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        accuracies.append(accuracy_score(y_test, pred))
        f1_scores.append(f1_score(y_test, pred, average='weighted'))
        
    best_acc_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1_scores)
        
    print(f"Best test Accuracy: {accuracies[best_acc_idx]:.4f} at C={C_values[best_acc_idx]}")
    print(f"Best F1_score: {f1_scores[best_f1_idx]:.4f} at C={C_values[best_f1_idx]}")