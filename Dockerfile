# PhishGuard web application container (CI/CD Stage 2).
# Builds a consistent image so the app runs the same on any machine.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
COPY webapp/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the project.
COPY . .

# The model artifacts are not committed. Generate them at build time. This step
# needs the dataset at data/phishing_nlp_dataset.csv in the build context; if it
# is absent the build continues and models/ can instead be mounted at runtime.
RUN python 20_train_multiclass.py \
    || echo "Model not built at image time. Provide data/phishing_nlp_dataset.csv or mount models/ when running."

EXPOSE 5000
WORKDIR /app/webapp

# Serve with gunicorn. The Flask application object is named app in app.py.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
