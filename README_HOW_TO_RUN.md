# Project 1: Customer Churn Prediction (how to run in VS Code)

## Which file to run
- Run **`main.py`**. It runs the whole project in order: it uses the other .py files for you.
- Do **not** run `Customer_Churn_Prediction.ipynb` with the `python` command. A notebook is not a normal Python file, and it gives the error "name 'null' is not defined". If you want the notebook, open it by clicking it in VS Code and use the Run buttons on the cells.

## 1. One-time setup
1. Install **Python 3.10 or newer** (python.org). Tick "Add Python to PATH".
2. Install **VS Code** with the **Python** extension (and the **Jupyter** extension if you want to use the notebook).
3. Open this project folder in VS Code (File > Open Folder), then open a terminal (Terminal > New Terminal) and run:
   ```
   pip install -r requirements.txt
   ```

## 2. The dataset
The dataset is already included in the **`data`** folder (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). It is the public Telco Customer Churn sample dataset (7,043 customers, 21 columns), the same data that is on Kaggle. You do not need to download anything.
In your README and report, mention the source: "Telco Customer Churn dataset (IBM sample data, available on Kaggle)".

## 3. Run the project
In the terminal, inside this folder:
```
python main.py
```
It takes about a minute. For a faster run, use `python main.py --skip-tuning`.

You will see in the terminal: the cleaning summary, the model comparison table (accuracy, precision, recall, F1, ROC-AUC), the classification report, cross-validation scores and the tuning result.
A new folder **`outputs`** is created with:
- `figures/`: all the EDA charts, the confusion matrix with ROC curves, and the feature-importance chart
- `model_results.csv`: the comparison table
- `churn_model.pkl`: the saved model

## 4. Take screenshots for your report
- the terminal showing the model comparison table
- 2 or 3 EDA charts from `outputs/figures`
- the confusion matrix with ROC curves, and the feature-importance chart

## 5. Write your notes
Open `MY_NOTES.md` and answer each question in your own words. Do not skip this: it is what you will explain in the viva.

## 6. Publish on GitHub and email Codec
1. Create a free account on github.com and make a new public repository.
2. Upload the .py files, the notebook, the `outputs/figures` charts, and the README (fill in the `{...}` parts with your own results). Link to the dataset page instead of uploading the CSV.
3. Email the repository link, together with your offer letter, to the address given in your Codec instructions, before the internship end date.

## 7. Be ready for the viva
- why the data was cleaned the way it was, and what one-hot encoding and scaling do
- why the train-test split is done before scaling (data leakage)
- why accuracy alone is not enough when only a minority leave (class imbalance, recall, ROC-AUC)
- what cross-validation and hyperparameter tuning do
- what the top features show, and what business action you would suggest

Read every file at least once and change something small (a different chart or model setting) so that you understand how it works.

## If something goes wrong
- **"pip is not recognized":** try `python -m pip install -r requirements.txt`, or reinstall Python with "Add Python to PATH" ticked.
- **"Dataset not found":** the CSV was moved or deleted from the `data` folder.
- **"ModuleNotFoundError":** run the pip install step again.
- Any other error: copy the red text and send it to me.

## After you finish
Send me your real results (metrics table, main insights, screenshots) and I will write the Project 1 part of your internship report from them.
