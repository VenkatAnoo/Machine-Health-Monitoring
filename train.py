import warnings
import joblib
import numpy as np
import pandas as pd

# Suppress sklearn parallel user warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def main():
  print("1. Loading dataset...")
  df = pd.read_csv("ai4i2020.csv")

  print("2. Preparing features and target...")
  X = df.drop(
      columns=[
          "UDI",
          "Product ID",
          "Torque [Nm]",
          "Machine failure",
          "TWF",
          "HDF",
          "PWF",
          "OSF",
          "RNF",
      ]
  )
  y = df["Torque [Nm]"]

  numerical_cols = [
      "Air temperature [K]",
      "Process temperature [K]",
      "Rotational speed [rpm]",
      "Tool wear [min]",
  ]
  categorical_cols = ["Type"]

  # 3. Preprocessing pipeline
  preprocessor = ColumnTransformer(
      transformers=[
          ("num", MinMaxScaler(), numerical_cols),
          ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
      ]
  )

  # 4. Train-test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  print("5. Defining models and hyperparameter grids...")
  models = {
      "Random Forest": (
          RandomForestRegressor(random_state=42, n_jobs=-1),
          {
              "model__n_estimators": [100, 200],
              "model__max_depth": [10, 20, None],
              "model__min_samples_split": [2, 5],
          },
      ),
      "Gradient Boosting": (
          GradientBoostingRegressor(random_state=42),
          {
              "model__n_estimators": [100, 200],
              "model__learning_rate": [0.05, 0.1],
              "model__max_depth": [3, 5],
          },
      ),
  }

  results = []
  best_models = {}

  print("6. Training and tuning models...")
  for name, (estimator, params) in models.items():
    print(f"--> Training {name}...")
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", estimator)]
    )

    grid = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    results.append(
        {
            "Model": name,
            "CV R2": grid.best_score_,
            "Test R2": r2,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        }
    )

  # 7. Model Comparison and Selection
  results_df = pd.DataFrame(results).sort_values(
      by="Test R2", ascending=False
  )
  print("\nMODEL COMPARISON:")
  print(results_df)

  best_model_name = results_df.iloc[0]["Model"]
  final_model = best_models[best_model_name]

  print(f"\nBest Overall Model: {best_model_name}")

  # 8. Save the final model
  model_filename = "final_torque_prediction_model.pkl"
  joblib.dump(final_model, model_filename)
  print(f"Final model saved successfully as '{model_filename}'!")


if __name__ == "__main__":
  main()