import os
import joblib

models_dir = "models"

for file in os.listdir(models_dir):
    if file.endswith(".pkl"):
        path = os.path.join(models_dir, file)

        try:
            model = joblib.load(path)

            print("=" * 50)
            print("File :", file)
            print("Type :", type(model).__name__)
            if hasattr(model, "get_params"):
              print(model.get_params())

        except Exception as e:
            print(f"Error loading {file}: {e}")