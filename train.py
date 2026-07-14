from src.preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model import train_all_models
from src.evaluate import evaluate_all_models

(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    df
) = preprocess_data("data/framingham.csv")


feature_engineering(df)


logistic_model, random_forest_model, best_model = train_all_models(

    X_train_scaled,

    X_train,

    y_train
)

evaluate_all_models(
    logistic_model,
    random_forest_model,
    best_model,
    X_test_scaled,
    X_test,
    y_test
)