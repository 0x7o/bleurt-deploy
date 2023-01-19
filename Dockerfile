FROM nvcr.io/nvidia/pytorch:22.10-py3

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Run the app
EXPOSE 5000
CMD ["python", "app.py"]