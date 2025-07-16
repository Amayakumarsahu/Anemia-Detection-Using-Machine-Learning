# Use Python 3.10 (compatible with your model)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run your Flask app with Gunicorn
CMD ["gunicorn", "app:app"]
