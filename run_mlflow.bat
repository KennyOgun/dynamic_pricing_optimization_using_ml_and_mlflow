@echo off
REM Start MLflow server with SQLite as backend store
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
pause
