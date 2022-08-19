from model import CatOrDog
from data_utils import DataLoader
from train import validate
import torch
import os


if __name__ == "__main__":
    test_loader = DataLoader(batch_size=256, path=os.path.join(os.getcwd(), "testing"))
    test_loader.load()
    for path in os.listdir("./"):
        if not path.startswith("model_training_data"):
            continue
        model = CatOrDog()
        print("Using state dict: ", path)
        model.load_state_dict(torch.load(path))
        test_acc, test_loss = validate(model, test_loader.data)
        print(f"Testing accuracy: {test_acc} | Testing loss: {test_loss}.")
        print("\n***************************************************************\n")
