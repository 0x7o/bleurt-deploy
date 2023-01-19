from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import fastapi

app = fastapi.FastAPI()

tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512")
model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512")
model.eval()
model.to("cuda")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(data: dict):
    references = data["references"]
    candidates = data["candidates"]
    with torch.no_grad():
        scores = (
            model(
                **tokenizer(
                    references, candidates, return_tensors="pt", padding=True
                ).to("cuda")
            )[0]
            .squeeze()
            .tolist()
        )
    return {"scores": scores}
