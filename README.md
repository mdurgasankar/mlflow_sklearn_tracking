<h1 align="center">MLflow Sklearn Tracking</h1>
<p align="center"><i>This repository contains a simple example of using MLflow for tracking and logging a scikit-learn model. The code demonstrates how to set up a linear model, log parameters, metrics, and the trained model using MLflow's tracking capabilities.</i></p>

## Prerequisites
    - Python 3.x
    - Docker (if running the code within a Docker environment)

## Setup

    1. Clone the repository:
        git clone https://github.com/mdurgasankar/mlflow_sklearn_tracking.git
        cd mlflow_sklearn_tracking
    2. Run the docker 
        docker-compose build 
        docker-compose up 
    3. To run the ML Model 
        docker ps   -> to get the mlflow_sklearn_tracking-model-training process container ID 
        docker exec -it <container ID > python elastic_net.py
    3. Access the MLflow UI:
        Open your web browser and navigate to http://localhost:5000 to view the MLflow UI.
    4. To stop the server 
        docker-compose down

## MLflow Tracking 
    Parameters logged:
        Regularization alpha
        L1 ratio
    Metrics logged:
        MAE
        r2
        RMSE

    Dynamic path based on model parameters and a timestamp

## MLflow UI
    - View and compare multiple runs.
    - Explore parameters, metrics, and artifacts logged during each run.








