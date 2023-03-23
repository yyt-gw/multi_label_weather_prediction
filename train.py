import argparse
import os
import yaml
import torch
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torchvision
from sklearn.metrics import classification_report
from datetime import datetime

from utils.loss_utils import criterion
from dataset.multi_label_weather_dataset import MultiLabelWeatherDataset
from model.model import MultiLabelBinaryClassifier


def train(model, device, lr_rate, epochs, train_loader, val_loader, steps):
    num_epochs = epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=steps, gamma=0.2
    )
    loss_func = torch.nn.BCELoss()
    train_epoch_loss = []
    validation_epoch_loss = []
    best_loss = 999.00
    best_model_wts = deepcopy(model.state_dict())
    model = model.to(device)
    for epoch in range(num_epochs):
        for data_loader in [
            {"type": "train", "loader": train_loader},
            {"type": "val", "loader": val_loader},
        ]:
            losses = []
            if data_loader["type"] == "train":
                model.train()
            else:
                model.eval()
            n_total_steps = len(data_loader["loader"])
            for i, samples in enumerate(data_loader["loader"]):
                images = samples["image_tensors"].to(device)
                gts = samples["labels"]
                scheduler.optimizer.zero_grad()
                with torch.set_grad_enabled(data_loader["type"] == "train"):
                    preds = model(images)
                    loss = criterion(loss_func, preds, gts, device=device)
                    losses.append(loss.item())
                    if data_loader["type"] == "train":
                        loss.backward()
                        scheduler.optimizer.step()
                current_loss = torch.tensor(losses).mean().item()
                print(
                    (
                        f'Epoch[{epoch+1}/{num_epochs}], Phase-> {data_loader["type"]},'
                        + f" Step[{i+1}/{n_total_steps}], Loss: {current_loss:.4f}"
                    )
                )
                if (i + 1) % (int(n_total_steps / 1)) == 0:
                    if data_loader["type"] == "val":
                        validation_epoch_loss.append(current_loss)
                        print(f"Current loss :{current_loss} \nBest loss :{best_loss}")
                        if current_loss < best_loss:
                            best_loss = current_loss
                            print("Updating checkpoint..")
                            best_model_wts = deepcopy(model.state_dict())
                    if data_loader["type"] == "train":
                        train_epoch_loss.append(current_loss)
    model.load_state_dict(best_model_wts)
    return model, zip(*[train_epoch_loss, validation_epoch_loss])


def evaluate(model, device, data_loader, classes):
    model.eval()
    weather_gts, weather_preds = {}, {}
    for class_ in classes:
        weather_gts[class_] = []
        weather_preds[class_] = []
    for i, samples in enumerate(data_loader):
        images = samples["image_tensors"].to(device)
        gts = samples["labels"]
        for weather, val in gts.items():
            weather_gts[weather] = weather_gts[weather] + [
                int(label) for label in val.tolist()
            ]
        with torch.no_grad():
            preds = model(images)
        for weather, val in preds.items():
            weather_preds[weather] = weather_preds[weather] + [
                round(prob[0]) for prob in val.tolist()
            ]
    for class_ in classes:
        print(weather_gts[class_][:8])
        print(weather_preds[class_][:8])
        print(f"Classification report for {class_} condition")
        print(
            classification_report(
                weather_gts[class_],
                weather_preds[class_],
                target_names=[f"Not {class_}", class_],
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/config.yml")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    with open(config["classes_path"], "r") as f:
        classes = f.read().rstrip().split("\n")
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_dataset = MultiLabelWeatherDataset(
        config["train_path"], config["train_images_root"], transforms, classes
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_dataset = MultiLabelWeatherDataset(
        config["val_path"], config["val_images_root"], transforms, classes
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    classifier = MultiLabelBinaryClassifier(classes, config["frozen_layers"], device)
    best_model, losses = train(
        classifier,
        device,
        config["lr"],
        config["epochs"],
        train_loader,
        val_loader,
        config["steps"],
    )
    exp_ext = datetime.now().strftime("%H_%M_%S")
    writer = SummaryWriter()
    for i, loss_info in enumerate(losses):
        writer.add_scalars("CE-Loss", {"train": loss_info[0], "val": loss_info[1]}, i)
    writer.flush()
    writer.close()

    weights_dir = f"./weights/exp_at_{exp_ext}/"
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(best_model.state_dict(), os.path.join(weights_dir, "best_model.pt"))

    if args.evaluate:
        print("best model evaluation...")
        evaluate(best_model, config["device"], val_loader, classes)
