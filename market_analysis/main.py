import argparse
from pathlib import Path
from sklearn.metrics import roc_curve

from src.data_cleaning import load_and_clean_data, missing_summary
from src.eda import asset_summary
from src.features import add_predictive_features, add_return_features, add_rolling_features, add_target
from src.market_model import add_rolling_beta, market_model_summary
from src.modeling import build_model_table, chronological_split, make_xy, train_logistic_regression
from src.evaluation import majority_baseline, coefficient_table, evaluate_classifier, prediction_table
from src.visualization import plot_confusion_matrix, plot_roc_curve


def run(data_path: str, output_dir: str = "outputs") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data(data_path)
    missing_summary(df).to_csv(output / "missing_summary.csv")
    asset_summary(df).to_csv(output / "asset_summary.csv", index=False)

    df = add_return_features(df)
    df = add_rolling_features(df)
    market_model_summary(df).to_csv(output / "market_model_summary.csv")
    df = add_rolling_beta(df)
    df = add_predictive_features(df)
    df = add_target(df)

    model_df = build_model_table(df)
    train_df, test_df, split_date = chronological_split(model_df)
    X_train, X_test, y_train, y_test = make_xy(train_df, test_df)

    majority_class, baseline_accuracy = majority_baseline(y_train, y_test)
    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)

    predictions = prediction_table(test_df, metrics["predictions"], metrics["probabilities"])
    predictions.to_csv(output / "test_predictions.csv", index=False)
    coefficient_table(model).to_csv(output / "logistic_coefficients.csv", index=False)

    plot_confusion_matrix(metrics["confusion_matrix"], output / "confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_test, metrics["probabilities"])
    plot_roc_curve(fpr, tpr, metrics["auc"], output / "roc_curve.png")

    print(f"Split date: {split_date}")
    print(f"Train observations: {len(train_df)}")
    print(f"Test observations: {len(test_df)}")
    print(f"Majority class: {majority_class}")
    print(f"Baseline accuracy: {baseline_accuracy:.3f}")
    print(f"Model accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC AUC: {metrics['auc']:.3f}")
    print("\nClassification report:\n")
    print(metrics["classification_report"])


def main():
    parser = argparse.ArgumentParser(description="market-analysis practice pipeline")
    parser.add_argument("data", help="Path to data_analysis_practice.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.data, args.output_dir)


if __name__ == "__main__":
    main()
