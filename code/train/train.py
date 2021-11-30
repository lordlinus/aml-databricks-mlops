import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("lightgbm")
install("azureml-sdk[databricks]")
install("azureml-mlflow")
install("joblib")

import time
from random import randint

import joblib
import lightgbm as lgb
import mlflow
import mlflow.azureml
import numpy as np
import pandas as pd
import sklearn
from azureml.core import Model, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.environment import CondaDependencies, Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.core.webservice import AksWebservice, Webservice
from azureml.core.webservice.aks import AksServiceDeploymentConfiguration
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from azureml.core.authentication import ServicePrincipalAuthentication

tenantId = dbutils.secrets.get(scope="key-vault-secrets", key="tenantId")
clientId = dbutils.secrets.get(scope="key-vault-secrets", key="clientId")
clientSecret = dbutils.secrets.get(scope="key-vault-secrets", key="clientSecret")

sp = ServicePrincipalAuthentication(
    tenant_id=tenantId,
    service_principal_id=clientId,
    service_principal_password=clientSecret,
)

subscription_id = "7c1d967f-37f1-4047-bef7-05af9aa80fe2"
resource_group = "demo-rg-01"
workspace_name = "aml-workspace-01"

workspace_region = "southeastasia"  # your region (if workspace need to be created)
experiment_name = "ss-lightgbm-exp"
model_name = "ss-lightgbm-model"

workspace = Workspace.get(
    name=workspace_name,
    location=workspace_region,
    resource_group=resource_group,
    subscription_id=subscription_id,
    auth=sp,
)

workspace.get_details()
mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
mlflow.set_experiment(experiment_name)


csv_file_path = "/dbfs/FileStore/data/Breast_cancer_data_0.csv"
dbfs_csv_file_path = "/FileStore/data/Breast_cancer_data_0.csv"


df = pd.read_csv(csv_file_path)
df.head()
df.info()


# define functions
def preprocess_data(df):
    X = df[
        [
            "mean_radius",
            "mean_texture",
            "mean_perimeter",
            "mean_area",
            "mean_smoothness",
        ]
    ]
    y = df["diagnosis"]

    enc = LabelEncoder()
    y = enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=16
    )

    return X_train, X_test, y_train, y_test, enc


def train_model(params, num_boost_round, X_train, X_test, y_train, y_test):
    t1 = time.time()
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[test_data],
        valid_names=["test"],
    )
    t2 = time.time()

    return model, t2 - t1


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict(X_test)
    y_pred = y_proba.argmax(axis=1)
    loss = log_loss(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    return loss, acc


# preprocess data
X_train, X_test, y_train, y_test, enc = preprocess_data(df)

# set training parameters
params = {
    "objective": "multiclass",
    "num_class": 2,
    "learning_rate": 0.1,
    "metric": "multi_logloss",
    "colsample_bytree": 1.0,
    "subsample": 1.0,
    "seed": 16,
}

num_boost_round = 32

with mlflow.start_run() as run:
    # enable automatic logging
    mlflow.lightgbm.autolog()

    # train model
    model, train_time = train_model(
        params, num_boost_round, X_train, X_test, y_train, y_test
    )
    mlflow.log_metric("training_time", train_time)

    # evaluate model
    loss, acc = evaluate_model(model, X_test, y_test)
    mlflow.log_metrics({"loss": loss, "accuracy": acc})

runs_df = mlflow.search_runs()
runs_df = runs_df.loc[runs_df["status"] == "FINISHED"]
runs_df = runs_df.sort_values(by="end_time", ascending=False)
print(runs_df.head())
run_id = runs_df.at[0, "run_id"]

joblib.dump(model, "model.pkl")

azureml_model = Model.register(
    workspace=workspace,
    model_name=model_name,  # Name of the registered model in your workspace.
    model_path="./model.pkl",  # Local file to upload and register as a model.
    model_framework=Model.Framework.SCIKITLEARN,  # Framework used to create the model.
    model_framework_version=sklearn.__version__,  # Version of scikit-learn used to create the model.
    resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
    description="Sample ML Model",
    tags={"area": "azureml", "type": "databricks notebook"},
)

# Return the run_id of the model
dbutils.notebook.exit(run_id)
