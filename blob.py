from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class Blob:
    def __init__(self):
        """The __init__ function is needed for initial preparation. It is started once during deployment."""
        self.tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512")
        self.model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512")
        self.model.eval()
        self.model.to("cuda")

    def predict(self, model_inputs: dict):
        """The predict function is called for every prediction."""
        references = model_inputs["references"]
        candidates = model_inputs["candidates"]
        with torch.no_grad():
            scores = (
                self.model(
                    **self.tokenizer(
                        references, candidates, return_tensors="pt", padding=True
                    ).to("cuda")
                )[0]
                .squeeze()
                .tolist()
            )
        return {"scores": scores}
