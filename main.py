"""
Customer Churn Prediction: run the whole project with one command.

    python main.py                 (full run, includes tuning)
    python main.py --skip-tuning   (faster run)

Charts are saved in outputs/figures, tables in outputs/, and the trained model in outputs/churn_model.pkl.
"""
import argparse

import matplotlib

matplotlib.use("Agg")          # save charts as files instead of opening windows

import joblib
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from config import OUTPUT_DIR
from data_preparation import clean_data, load_data
from eda import run_eda
from evaluation import evaluate_models, plot_confusion_and_roc, plot_feature_importance, print_report
from modeling import (build_models, build_preprocessor, cross_validate_models, split_data, train_models,
                      tune_random_forest)


def main(skip_tuning=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, "\nSTEP 1: Load and clean the data\n" + "=" * 60)
    df = clean_data(load_data())

    print("\n" + "=" * 60, "\nSTEP 2: Exploratory data analysis\n" + "=" * 60)
    run_eda(df)

    print("\n" + "=" * 60, "\nSTEP 3: Prepare the data and train the models\n" + "=" * 60)
    X_train, X_test, y_train, y_test = split_data(df)
    print("Train size:", X_train.shape, "| Test size:", X_test.shape)
    pre = build_preprocessor(X_train)
    models = train_models(build_models(pre), X_train, y_train)

    print("\n" + "=" * 60, "\nSTEP 4: Evaluate the models\n" + "=" * 60)
    results = evaluate_models(models, X_test, y_test)
    print(results)
    results.to_csv(OUTPUT_DIR / "model_results.csv")
    best_name = results["ROC-AUC"].idxmax()
    print("\nBest model by ROC-AUC:", best_name)
    print_report(models[best_name], X_test, y_test)
    plot_confusion_and_roc(models, best_name, X_test, y_test)

    print("\n" + "=" * 60, "\nSTEP 5: Cross-validation\n" + "=" * 60)
    print(cross_validate_models(models, X_train, y_train))

    final_model = models[best_name]
    if not skip_tuning:
        print("\n" + "=" * 60, "\nSTEP 6: Hyperparameter tuning (Random Forest)\n" + "=" * 60)
        search = tune_random_forest(models["Random Forest"], X_train, y_train)
        tuned = search.best_estimator_
        print("Best parameters:", search.best_params_)
        print("Best cross-validated ROC-AUC:", round(search.best_score_, 3))
        pred = tuned.predict(X_test)
        proba = tuned.predict_proba(X_test)[:, 1]
        print("Tuned Random Forest on the test set:")
        print("  Accuracy:", round(accuracy_score(y_test, pred), 3))
        print("  Recall  :", round(recall_score(y_test, pred), 3))
        print("  ROC-AUC :", round(roc_auc_score(y_test, proba), 3))

        print("\n" + "=" * 60, "\nSTEP 7: Which features drive churn?\n" + "=" * 60)
        print(plot_feature_importance(tuned).round(3))
        if best_name == "Random Forest":
            final_model = tuned

    joblib.dump(final_model, OUTPUT_DIR / "churn_model.pkl")
    print("\nDone. Check the 'outputs' folder for charts, results and the saved model.")
    print("Now write your own observations in MY_NOTES.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tuning", action="store_true", help="skip hyperparameter tuning to run faster")
    main(parser.parse_args().skip_tuning)
