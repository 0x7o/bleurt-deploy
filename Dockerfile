FROM nvcr.io/nvidia/pytorch:22.10-py3

# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Run the app using uvicorn
EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]