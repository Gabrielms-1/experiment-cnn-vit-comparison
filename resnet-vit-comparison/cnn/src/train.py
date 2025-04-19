import torch
import os
import yaml
import torch.nn as nn
from dataset import FolderBasedDataset, create_dataloader
from resnet50 import ResNet50
import wandb
import argparse
from datetime import datetime
from importlib.machinery import SourceFileLoader
import io
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

utils_seed_path = "./resnet-vit-comparison/utils/__init__.py"
seed_module = SourceFileLoader("utils_seed", utils_seed_path).load_module()

def process_data(train_dataset_path, val_dataset_path, resize, batch_size, seed):
    train_dataset = FolderBasedDataset(train_dataset_path, resize)
    val_dataset = FolderBasedDataset(val_dataset_path, resize)
    
    train_loader, val_loader = create_dataloader(train_dataset, val_dataset, batch_size, seed)

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
    
def train_model(model, total_epochs, optimizer, criterion, train_loader, val_loader, device, n_classes, scheduler_step, scheduler_factor):
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []

    best_f1_score = 0
    f1_patience = 5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_factor)
    
    for epoch in range(total_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        model.train()
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

        f1_score = torch.mean(val_f1_score)
        wandb.log({
            "epoch": epoch+1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "train_accuracy": epoch_accuracy,
            "val_accuracy": val_accuracy,
            "f1_score": f1_score,
            "lr": optimizer.param_groups[0]['lr']
        })

        print("-" * 50)
        print(f"EPOCH: {epoch+1}")
        print(f"- train_loss: {epoch_loss:.4f} | train_accuracy: {epoch_accuracy:.4f}")
        print(f"- val_loss: {val_loss:.4f} | val_accuracy: {val_accuracy:.4f} | f1_score: {f1_score:.4f}")
        print(f"- learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        print("-" * 50) 

        if f1_score >= best_f1_score:
            best_f1_score = f1_score
            f1_patience = 5
            torch.save(model.state_dict(), f"{args.check_point_dir}/model_checkpoint_best_f1_score.pth")
            print(f"Model saved at epoch {epoch+1}")
        else:
            f1_patience -= 1
            if f1_patience <= 0:
                break
        if f1_score >= 0.93:
            if val_accuracy >= 0.93 and val_loss <= 0.24:
                torch.save(model.state_dict(), f"{args.check_point_dir}/model_checkpoint_{epoch+1}.pth")
                print(f"Model saved at epoch {epoch+1}")
                break
        scheduler.step()

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
            "dropout": args.dropout,
            "scheduler_step": args.scheduler_step,
            "scheduler_factor": args.scheduler_factor,
        },
    )
    os.makedirs(args.check_point_dir, exist_ok=True)

    train_loader, val_loader = process_data(args.train_dir, args.val_dir, args.resize, args.batch_size, args.seed)
    val_dataset = val_loader.dataset
    model = ResNet50(args.n_classes, args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses, val_losses, val_accuracies, train_accuracies, confusion_matrix = train_model(model, args.epochs, optimizer, criterion, train_loader, val_loader, device, args.n_classes, args.scheduler_step, args.scheduler_factor)

    class_names = [str(val_dataset.int_to_label_map[i]) for i in range(confusion_matrix.shape[0])]

    cm_numpy = confusion_matrix.cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_numpy, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    cm_image = Image.open(buf)

    wandb.log({"confusion_matrix_image": wandb.Image(cm_image)})

    wandb.finish()

    save_path = os.path.join(args.check_point_dir, "model_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    with open(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'train.yaml'))}", "r") as f:
        config = yaml.safe_load(f)
       
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default=config["CNN"]["PROJECT"]["name"])
    parser.add_argument("--train_dir", type=str, default=config["LOCAL"]["train_dir"])
    parser.add_argument("--val_dir", type=str, default=config["LOCAL"]["val_dir"])
    parser.add_argument("--epochs", type=int, default=config["TRAIN"]["epochs"])
    parser.add_argument("--batch_size", type=int, default=config["TRAIN"]["batch_size"])
    parser.add_argument("--lr", type=float, default=config["CNN"]["MODEL"]["learning_rate"])
    parser.add_argument("--weight_decay", type=float, default=config["CNN"]["MODEL"]["weight_decay"])
    parser.add_argument("--dropout", type=float, default=config["CNN"]["MODEL"]["dropout"])
    parser.add_argument("--scheduler_step", type=int, default=config["CNN"]["MODEL"]["scheduler_step"])
    parser.add_argument("--scheduler_factor", type=float, default=config["CNN"]["MODEL"]["scheduler_factor"])
    parser.add_argument("--resize", type=int, default=config["TRAIN"]["img_size"])
    parser.add_argument("--timestamp", type=str, default=datetime.now().strftime("%Y%m%d-%H-%M"))
    parser.add_argument("--n_classes", type=int, default=config["TRAIN"]["n_classes"])
    parser.add_argument("--check_point_dir", type=str, default=config["LOCAL"]["check_point_dir"])
    parser.add_argument("--seed", type=int, default=config["TRAIN"]["seed"])
    args = parser.parse_args()
    
    seed_module.set_seed(args.seed)

    main(args)