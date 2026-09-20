"""Classification baselines and evaluation helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from .config import FEATURES, TARGET


def majority_baseline(y_train, y_test) -> tuple[int, float]:
    majority_class = int(y_train.mode()[0])
    predictions = np.full(len(y_test), majority_class)
    return majority_class, accuracy_score(y_test, predictions)


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "predictions": y_pred,
        "probabilities": y_prob,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=3),
    }


def coefficient_table(model) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    result = pd.DataFrame({"feature": FEATURES, "coefficient": classifier.coef_[0]})
    result["abs_coefficient"] = result["coefficient"].abs()
    return result.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def prediction_table(test_df: pd.DataFrame, predictions, probabilities) -> pd.DataFrame:
    result = test_df[["date", "asset", TARGET, "return_1d"]].copy()
    result["prediction"] = predictions
    result["prob_positive"] = probabilities
    return result
