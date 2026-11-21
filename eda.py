"""Step 2: exploratory data analysis. Every chart is saved as a PNG in outputs/figures."""
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIG_DIR

sns.set_theme(style="whitegrid")


def _save(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()
    print("  saved:", FIG_DIR.name + "/" + name)


def churn_rate_plot(df, col):
    rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False) * 100
    rate.plot(kind="bar", figsize=(6, 4), color="#2E6F9E")
    plt.ylabel("Churn rate (%)")
    plt.title(f"Churn rate by {col}")
    plt.xticks(rotation=30, ha="right")
    _save(f"eda_churn_rate_by_{col}.png")


def run_eda(df):
    print("\n[EDA] Creating charts...")

    sns.countplot(x="Churn", data=df)
    plt.title("Customers who churned (1) and stayed (0)")
    _save("eda_churn_count.png")

    for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
        churn_rate_plot(df, col)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack", ax=axes[0])
    axes[0].set_title("Tenure (months) by churn")
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1])
    axes[1].set_title("Monthly charges by churn")
    _save("eda_tenure_and_charges.png")

    sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation of numerical columns")
    _save("eda_correlation_heatmap.png")
