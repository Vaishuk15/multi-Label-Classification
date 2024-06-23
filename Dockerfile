# FROM python:3.9

# WORKDIR /code

# COPY . .

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# CMD ["python", "fast_api.py", "--port", "8000"]


# Use official Python runtime as a parent image
FROM python:3.10
 
 
# Set the working directory in the container
WORKDIR /app
 
# Install Python dependencies
COPY . .
 
RUN pip install -r requirements.txt
 
# Copy the application code into the container
 
# Expose port 8000
EXPOSE 8000
 
# Command to run the FastAPI application with Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
