# This file defines the Docker container that will contain the Flask app.

# From the source image
FROM python:3.6-slim

# Set the default working directory
WORKDIR /app/

# Copy requirements.txt outside the container
# to /app/ inside the container
COPY requirements.txt /app/

# Install required packages
RUN pip install -r ./requirements.txt

# Copy app.py and__init__.py outside the container
# to /app/ inside the container
COPY app.py __init__.py /app/

# Copy model.pkl outside the container
# to /app/ inside the container
COPY model_zad_1.pkl /app/

# Expose the container's port 5000
EXPOSE 5000

# When the container starts, run this
ENTRYPOINT python ./app.py