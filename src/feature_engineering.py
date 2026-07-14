import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier


# Create folders if they don't exist
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def correlation_heatmap(df):
    """
    Generate and save correlation heatmap.
    """

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("images/correlation_heatmap.png")

    plt.close()

    print("✔ Correlation Heatmap Saved")


def top_correlated_features(df, target="TenYearCHD"):
    """
    Find top correlated features with target.
    """

    correlation = df.corr()[target].sort_values(
        ascending=False
    )

    correlation = correlation.drop(target)

    top_features = correlation.to_frame(
        name="Correlation"
    )

    top_features.to_csv(
        "reports/top_correlated_features.csv"
    )

    print("\nTop Correlated Features:\n")
    print(top_features)

    return top_features


def feature_importance(X, y):
    """
    Train Random Forest and calculate feature importance.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.to_csv(
        "reports/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 7))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance
    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        "images/feature_importance.png"
    )

    plt.close()

    print("✔ Feature Importance Graph Saved")

    return importance


def plot_target_distribution(df, target="TenYearCHD"):
    """
    Plot target class distribution.
    """

    plt.figure(figsize=(6, 5))

    sns.countplot(
        x=target,
        data=df
    )

    plt.title("Target Distribution")

    plt.tight_layout()

    plt.savefig(
        "images/target_distribution.png"
    )

    plt.close()

    print("✔ Target Distribution Saved")


def feature_engineering(df, target="TenYearCHD"):
    """
    Complete Feature Engineering Pipeline.
    """

    print("\nGenerating Target Distribution...")
    plot_target_distribution(df, target)

    print("\nGenerating Correlation Heatmap...")
    correlation_heatmap(df)

    print("\nFinding Top Correlated Features...")
    top_corr = top_correlated_features(df, target)

    print("\nGenerating Feature Importance...")
    X = df.drop(columns=[target])
    y = df[target]

    importance = feature_importance(X, y)

    print("\nFeature Engineering Completed Successfully!")

    return top_corr, importance