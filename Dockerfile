<<<<<<< HEAD
# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
#RUN python -m playwright install chromium

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]
=======
FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
>>>>>>> 500dd903497dd5c2a1a7b7d60deff3813e4ffdf5
