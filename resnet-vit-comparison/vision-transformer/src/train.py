import argparse
import yaml
import os
import torch
from dataset import FolderBasedDataset, create_dataloader
from vision_transformer import VisionTransformer
import wandb
from datetime import datetime
import io
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader

utils_seed_path = "./resnet-vit-comparison/utils/__init__.py"
seed_module = SourceFileLoader("utils_seed", utils_seed_path).load_module()

def process_data(train_dataset, val_dataset, resize, batch_size, seed):

    train_dataset = FolderBasedDataset(train_dataset, resize)
    val_dataset = FolderBasedDataset(val_dataset, resize)
    
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


def train_model(model, total_epochs, optimizer, criterion, train_loader, val_loader, device, n_classes):

    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []    

    for epoch in range(total_epochs):
        model.train()

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
        print(f"- val_loss: {val_loss:.4f} | val_accuracy: {val_accuracy:.4f} | f1_score: {f1_score:.4f}")
        print(f"-" * 50)

    return train_losses, val_losses, val_accuracies, train_accuracies, confusion_matrix

def main(args):

    wandb.init(
        project=f"{args.project_name}",
        name=f"{args.project_name}-{args.timestamp}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "d_model": args.d_model,
            "n_classes": args.n_classes,
            "img_size": args.img_size,
            "patch_size": args.patch_size,
            "n_channels": args.n_channels,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "weight_decay": args.weight_decay,
        },
    )

    os.makedirs(args.check_point_dir, exist_ok=True)

    train_loader, val_loader = process_data(args.train_dir, args.val_dir, args.resize, args.batch_size, args.seed)
    val_dataset = FolderBasedDataset(args.val_dir, args.resize)

    model = VisionTransformer(
        args.d_model,
        args.n_classes,
        args.img_size,
        args.patch_size,
        args.n_channels,
        args.n_heads,
        args.n_layers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_losses, val_losses, val_accuracies, train_accuracies, confusion_matrix = train_model(model, args.epochs, optimizer, criterion, train_loader, val_loader, device, args.n_classes)

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

    return

if __name__ == "__main__":
    with open(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'train.yaml'))}", "r") as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=config["LOCAL"]["train_dir"])
    parser.add_argument("--val_dir", type=str, default=config["LOCAL"]["val_dir"])
    parser.add_argument("--batch_size", type=int, default=config["TRAIN"]["batch_size"])
    parser.add_argument("--resize", type=int, default=config["TRAIN"]["img_size"])
    parser.add_argument("--timestamp", type=str, default=datetime.now().strftime("%Y%m%d-%H-%M"))
    parser.add_argument("--n_classes", type=int, default=config["TRAIN"]["n_classes"])
    parser.add_argument("--project_name", type=str, default=config["VIT"]["PROJECT"]["name"])
    parser.add_argument("--check_point_dir", type=str, default=config["LOCAL"]["check_point_dir"])
    parser.add_argument("--epochs", type=int, default=config["TRAIN"]["epochs"])
    parser.add_argument("--lr", type=float, default=config["TRAIN"]["lr"])
    parser.add_argument("--d_model", type=int, default=config["VIT"]["MODEL"]["d_model"])
    parser.add_argument("--img_size", type=int, default=config["TRAIN"]["img_size"])
    parser.add_argument("--patch_size", type=int, default=config["VIT"]["MODEL"]["patch_size"])
    parser.add_argument("--n_channels", type=int, default=config["VIT"]["MODEL"]["n_channels"])
    parser.add_argument("--n_heads", type=int, default=config["VIT"]["MODEL"]["n_heads"])
    parser.add_argument("--n_layers", type=int, default=config["VIT"]["MODEL"]["n_layers"])
    parser.add_argument("--weight_decay", type=float, default=config["TRAIN"]["weight_decay"])
    parser.add_argument("--seed", type=int, default=config["TRAIN"]["seed"])
    args = parser.parse_args()
    
    seed_module.set_seed(args.seed)

    main(args)
