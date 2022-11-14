"""Evaluation script for measuring mean squared error."""
import os
import json
import subprocess
import sys
import numpy as np
import pathlib
import tarfile
import json
import logging
import requests
from PIL import Image


test_dir = '/opt/ml/processing'
n_classes = 3
model_path = f"/opt/ml/processing/model/model.tar.gz"


with tarfile.open(model_path, "r:gz") as tar:
    tar.extractall("./model")
     
        
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def eval_model(model, data_loader, device, n_examples):

    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, dim=1)

            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / n_examples

if __name__ == "__main__":
    print("starting in main")
    
    install("torch==1.8")
    install("torchvision==0.9")

    
    import torch
    from torchvision import transforms
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from torch import nn
    from torchvision import models

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet34(pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)
    model.load_state_dict(torch.load("./model/model.pth", map_location=torch.device('cpu')))
    model.eval()   

    DATA_DIR = test_dir


    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    transforms = {'train': T.Compose([
        T.RandomResizedCrop(size=256),
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ]), 'test': T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ]),
    }


    DATASETS = ['test']

    image_datasets = {d: ImageFolder(f'{DATA_DIR}/{d}', transforms[d]) for d in DATASETS}

    data_loaders = {d: DataLoader(image_datasets[d], batch_size=8, shuffle=True, num_workers=4) for d in DATASETS}
    
    dataset_sizes = {d: len(image_datasets[d]) for d in DATASETS}

    val_acc = eval_model(
            model,
            data_loaders['test'],
            device,
            dataset_sizes['test']
        )

    print(f'accuracy {val_acc}')
    
    report_dict = {
        "Test_set_metrics": {
                "accuracy": val_acc.tolist()
            }
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))