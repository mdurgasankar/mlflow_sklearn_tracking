# train.py
import os
import warnings
import sys
import datetime
 
import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

 
# Import mlflow
import mlflow
import mlflow.sklearn


def load_data():
    # Load diabetes dataset
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    # Create pandas DataFrame 
    Y = np.array([y]).transpose()
    d = np.concatenate((X, Y), axis=1)
    cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
    data = pd.DataFrame(d, columns=cols)
    return data, X, y


def train_model(in_alpha, in_l1_ratio):
    data, X, y = load_data()

    # Evaluate metrics
    def eval_metrics(actual, pred):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mae = mean_absolute_error(actual, pred)
      r2 = r2_score(actual, pred)
      return rmse, mae, r2

    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    
    # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
    train_x = train.drop(["progression"], axis=1)
    test_x = test.drop(["progression"], axis=1)
    train_y = train[["progression"]]
    test_y = test[["progression"]]
    
    if float(in_alpha) is None:
        alpha = 0.05
    else:
        alpha = float(in_alpha)
        
    if float(in_l1_ratio) is None:
        l1_ratio = 0.05
    else:
        l1_ratio = float(in_l1_ratio)

    mlflow.set_tracking_uri("http://mlflow-ui:5000")  # Replace with your MLflow tracking server URI
    mlflow.set_experiment("ElasticNet-trail1")

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
    
        predicted_qualities = lr.predict(test_x)
    
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    
        # Print out ElasticNet model metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
    
        # Log mlflow attributes for mlflow UI
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "model")

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        modelpath = "./test_diabetes/model-%f-%f-%s" % (alpha, l1_ratio, timestamp)
        mlflow.sklearn.save_model(lr, modelpath)


if __name__ == "__main__":
    train_model(0.01, 0.01)
