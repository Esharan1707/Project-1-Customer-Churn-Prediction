# Customer Churn Prediction

Predicts which telecom customers are likely to stop using the service and identifies the factors behind churn.

## Dataset
Telco Customer Churn dataset (IBM sample data, available on Kaggle). A copy is in the `data` folder.

## Project structure
| File | What it does |
|---|---|
| `main.py` | Runs the whole project: `python main.py` |
| `config.py` | Paths and settings |
| `data_preparation.py` | Loads and cleans the data |
| `eda.py` | Creates the EDA charts |
| `modeling.py` | Prepares features, builds and trains the models, cross-validation and tuning |
| `evaluation.py` | Metrics, confusion matrix, ROC curves, feature importance |
| `Customer_Churn_Prediction.ipynb` | The same project as a notebook, with explanations |

## What I did
- Cleaned the data: {describe what you cleaned}
- Explored the data (EDA): {one line about your main findings}
- Built models: Logistic Regression, Random Forest and XGBoost
- Evaluated with accuracy, recall and ROC-AUC, plus cross-validation and tuning

## Results
| Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| {fill from outputs/model_results.csv} | | | |

## Key insights
- {your insight 1}
- {your insight 2}

## How to run
```
pip install -r requirements.txt
python main.py
```
The dataset is already in the `data` folder.

Built during the Data Analytics Internship at Codec Technologies by Ekanshi Sharan Srivastava.
