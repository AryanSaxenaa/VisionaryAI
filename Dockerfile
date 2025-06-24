# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the main working directory
WORKDIR /app

# Copy the requirements file and install dependencies
# This is done first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./backend /app/backend
COPY ./frontend /app/frontend
COPY ./models /app/models

# ---- FIX IS HERE ----
# Change the working directory to where our app lives
WORKDIR /app/backend

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the application using Uvicorn
# The command is now simpler because we are in the correct directory
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]