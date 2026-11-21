"""Step 3: prepare features, build and train the models, cross-validate and tune."""
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CV_FOLDS, RANDOM_STATE, TARGET, TEST_SIZE

try:
    from xgboost import XGBClassifier      # optional: pip install xgboost
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def split_data(df):
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    # stratify keeps the same churn share in the train and test sets
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def build_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    print("Numerical columns :", num_cols)
    print("Categorical columns:", cat_cols)
    # Scale the numbers and one-hot encode the categories. The split is done first to avoid data leakage.
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def build_models(pre):
    """Churners are the minority class, so class_weight='balanced' gives them more importance."""
    models = {
        "Logistic Regression": Pipeline([("pre", clone(pre)),
                                         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("pre", clone(pre)),
                                   ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                                  random_state=RANDOM_STATE))]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([("pre", clone(pre)),
                                      ("clf", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                                            eval_metric="logloss", random_state=RANDOM_STATE))])
    else:
        print("XGBoost is not installed, so it is skipped (pip install xgboost).")
    return models


def train_models(models, X_train, y_train):
    for name, model in models.items():
        print(f"  training {name} ...")
        model.fit(X_train, y_train)
    return models


def cross_validate_models(models, X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        rows.append({"Model": name, "Mean ROC-AUC": scores.mean(), "Std": scores.std()})
    return pd.DataFrame(rows).set_index("Model").round(3)


def tune_random_forest(rf_pipeline, X_train, y_train):
    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 6, 10, 15],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    search = RandomizedSearchCV(rf_pipeline, param_dist, n_iter=10, cv=3, scoring="roc_auc",
                                random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_train, y_train)
    return search
