# Bleurt Model API

This project is a REST API that allows you to use the pre-trained "Elron/bleurt-large-512" model to predict the similarity scores between a set of reference and candidate sentences. The API is built using the FastAPI framework and the Hugging Face Transformers library.

## Requirements

- Python 3.6 or higher
- PyTorch 1.5 or higher
- FastAPI
- Transformers

## Docker

You can run the API using Docker. You must have Docker installed on your machine to do this.

First, build the Docker image:

```bash
docker build -t bleurt-api .
```

Then, run the Docker container:

```bash
docker run -d -p 5000:5000 --gpus=all bleurt-api
```

Test the API by sending a POST request to http://localhost:8000/predict with a JSON payload containing the references and candidates.

## Example

```json
{
  "references": ["This is a great product", "This is a terrible product"],
  "candidates": ["This is a fantastic product", "This is a horrible product"]
}
```

```json
{
  "scores": [0.9656828045845032, 0.04987005889415741]
}
```

## Note

Make sure that the GPU is available if you are running the API in a container. If you are running the API on a machine without a GPU, you can remove the line `model.to("cuda")` from the code.
