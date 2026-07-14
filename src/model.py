import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score


# Create models folder if not exists
os.makedirs("models", exist_ok=True)


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model
    """

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "models/logistic_regression.pkl"
    )

    print("✔ Logistic Regression Saved")

    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest model
    """

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        "models/random_forest.pkl"
    )

    print("✔ Random Forest Saved")

    return model


def hyperparameter_tuning(X_train, y_train):
    """
    Tune Random Forest using GridSearchCV
    """

    print("\nRunning Hyperparameter Tuning...")

    params = {

        "n_estimators": [100, 200],

        "max_depth": [5, 10, None],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(

        estimator=RandomForestClassifier(
            random_state=42
        ),

        param_grid=params,

        cv=5,

        scoring="accuracy",

        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nBest Accuracy:")
    print(grid.best_score_)

    best_model = grid.best_estimator_

    joblib.dump(
        best_model,
        "models/best_model.pkl"
    )

    print("✔ Best Model Saved")

    return best_model


def cross_validation(model, X, y):
    """
    Perform 5-Fold Cross Validation
    """

    print("\nRunning Cross Validation...")

    scores = cross_val_score(

        model,

        X,

        y,

        cv=5,

        scoring="accuracy"
    )

    print("Cross Validation Scores:")
    print(scores)

    print("Average Accuracy:")
    print(scores.mean())

    return scores


def train_all_models(
        X_train_scaled,
        X_train,
        y_train
):
    """
    Train all models.
    """

    logistic = train_logistic_regression(
        X_train_scaled,
        y_train
    )

    random_forest = train_random_forest(
        X_train,
        y_train
    )

    best_model = hyperparameter_tuning(
        X_train,
        y_train
    )

    cross_validation(
        best_model,
        X_train,
        y_train
    )

    print("\nAll Models Trained Successfully!")

    return logistic, random_forest, best_model