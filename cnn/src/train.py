import torch
import os
import yaml
import torch.nn as nn
from dataset import FolderBasedDataset, create_dataloader
from resnet50 import ResNet50
import wandb
import argparse
from datetime import datetime

def process_data(train_dataset, val_dataset, resize, batch_size):
    train_dataset = FolderBasedDataset(train_dataset, resize)
    val_dataset = FolderBasedDataset(val_dataset, resize)
    
    train_loader, val_loader = create_dataloader(train_dataset, val_dataset, batch_size)

    return train_loader, val_loader

def compute_metrics(confusion_matrix):
    precision = torch.diag(confusion_matrix) / (torch.sum(confusion_matrix, dim=1) + 1e-10)
    recall = torch.diag(confusion_matrix) / (torch.sum(confusion_matrix, dim=0) + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    
    return precision, recall, f1_score

def evaluate_model(model, val_loader, criterion, device, n_classes):
    model.eval()
    
    val_loss = 0
    correct_predictions = 0
    total_samples = 0

    confusion_matrix = torch.zeros(n_classes, n_classes)
    
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            _, predicted = torch.max(output.detach(), 1)    
            loss = criterion(output, target)
            
            val_loss += loss.item() * data.size(0)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            for p, t in zip(predicted, target):
                confusion_matrix[t.long(), p.long()] += 1

        average_loss = val_loss / len(val_loader.dataset)
        accuracy = correct_predictions / total_samples

    precision, recall, f1_score = compute_metrics(confusion_matrix)

    return average_loss, accuracy, precision, recall, f1_score, confusion_matrix
    
def train_model(model, total_epochs, optimizer, criterion, train_loader, val_loader, device, n_classes):
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []

    model.train()
    
    for epoch in range(total_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)
            _, predicted = torch.max(output.detach(), 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * data.size(0)

            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)

        val_loss, val_accuracy, val_precision, val_recall, val_f1_score, confusion_matrix = evaluate_model(model, val_loader, criterion, device, n_classes)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        epoch_accuracy = correct_predictions / total_samples
        train_accuracies.append(epoch_accuracy)

        wandb.log({
            "epoch": epoch+1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "train_accuracy": epoch_accuracy,
            "val_accuracy": val_accuracy,
            "precision": val_precision,
            "recall": val_recall,
            "f1_score": val_f1_score,
        })

        f1_score = torch.mean(val_f1_score)

        print("-" * 50)
        print(f"EPOCH: {epoch+1}")
        print(f"- train_loss: {epoch_loss:.4f} | train_accuracy: {epoch_accuracy:.4f}")
        print(f"- val_loss: {val_loss:.4f} | val_accuracy: {val_accuracy:.4f}")
        print(f"- precision: {val_precision:.4f} | recall: {val_recall:.4f} | f1_score: {f1_score:.4f}")
        print("-" * 50) 

    return train_losses, val_losses, val_accuracies, train_accuracies, confusion_matrix

def main(args):
    wandb.init(
        project=f"{args.project_name}", 
        name=f"{args.project_name}-{args.timestamp}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "n_classes": args.n_classes,
            "img_size": args.resize,
            "weight_decay": args.weight_decay,
        },
    )
    os.makedirs(args.check_point_dir, exist_ok=True)

    train_loader, val_loader = process_data(args.train_dir, args.val_dir, args.resize, args.batch_size)
    
    model = ResNet50(args.n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses, val_losses, val_accuracies, train_accuracies, confusion_matrix = train_model(model, args.epochs, optimizer, criterion, train_loader, val_loader, device, args.n_classes)

if __name__ == "__main__":
    with open("config/train.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default=config["PROJECT"]["name"])
    parser.add_argument("--train_dir", type=str, default=config["LOCAL"]["train_dir"])
    parser.add_argument("--val_dir", type=str, default=config["LOCAL"]["val_dir"])
    parser.add_argument("--epochs", type=int, default=config["TRAIN"]["epochs"])
    parser.add_argument("--batch_size", type=int, default=config["TRAIN"]["batch_size"])
    parser.add_argument("--lr", type=float, default=config["TRAIN"]["lr"])
    parser.add_argument("--weight_decay", type=float, default=config["TRAIN"]["weight_decay"])
    parser.add_argument("--resize", type=int, default=config["MODEL"]["img_size"])
    parser.add_argument("--timestamp", type=str, default=datetime.now().strftime("%Y%m%d-%H-%M"))
    parser.add_argument("--n_classes", type=int, default=config["MODEL"]["n_classes"])
    parser.add_argument("--check_point_dir", type=str, default=config["LOCAL"]["check_point_dir"])
    args = parser.parse_args()
    
    main(args)