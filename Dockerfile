# Dockerfile
FROM python:3.8

# Set the working directory
WORKDIR /app


COPY requirements.txt .

RUN pip install -r requirements.txt


# Expose the default MLflow UI port
EXPOSE 5000

COPY src/ .

# Command to run MLflow UI
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]














# # Dockerfile
# FROM python:3.8

# WORKDIR /app

# RUN pip install mlflow


# ENV PATH="/usr/local/bin:${PATH}"


# COPY requirements.txt .

# RUN pip install -r requirements.txt


# COPY src/ .

# CMD ["mlflow", "ui"]
