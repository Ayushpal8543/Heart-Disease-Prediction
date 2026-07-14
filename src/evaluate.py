import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

# Create folders
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n========== {model_name} ==========")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1


def plot_confusion_matrix(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        f"images/confusion_matrix_{model_name.lower().replace(' ','_')}.png"
    )

    plt.show()


def plot_roc_curve(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )

    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig("images/roc_curve.png")

    plt.show()


def plot_precision_recall(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:,1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        y_prob
    )

    plt.figure(figsize=(7,6))

    plt.plot(
        recall,
        precision
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")

    plt.tight_layout()

    plt.savefig(
        "images/precision_recall_curve.png"
    )

    plt.show()


def compare_models(results):

    df = pd.DataFrame(results)

    df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    print("\nModel Comparison\n")

    print(df)

    return df


def evaluate_all_models(
    logistic_model,
    random_forest_model,
    best_model,
    X_test_scaled,
    X_test,
    y_test
):

    results = []

    # Logistic Regression
    acc, pre, rec, f1 = evaluate_model(
        logistic_model,
        X_test_scaled,
        y_test,
        "Logistic Regression"
    )

    plot_confusion_matrix(
        logistic_model,
        X_test_scaled,
        y_test,
        "Logistic Regression"
    )

    results.append({

        "Model":"Logistic Regression",

        "Accuracy":acc,

        "Precision":pre,

        "Recall":rec,

        "F1 Score":f1

    })

    # Random Forest
    acc, pre, rec, f1 = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )

    plot_confusion_matrix(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )

    results.append({

        "Model":"Random Forest",

        "Accuracy":acc,

        "Precision":pre,

        "Recall":rec,

        "F1 Score":f1

    })

    # Best Model
    plot_roc_curve(
        best_model,
        X_test,
        y_test
    )

    plot_precision_recall(
        best_model,
        X_test,
        y_test
    )

    compare_models(results)

    print("\nEvaluation Completed Successfully!")