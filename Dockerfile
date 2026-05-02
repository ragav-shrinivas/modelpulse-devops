FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port (Render uses PORT env, but keep 5000 for Flask)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]