# Use Python 3.10 as the base image
FROM python:3.10-slim

# Install Node.js (needed to build the React frontend)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Build the React frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Return to the root directory
WORKDIR /app

# Hugging Face Spaces requires apps to run on port 7860
EXPOSE 7860

# Start the FastAPI backend
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
