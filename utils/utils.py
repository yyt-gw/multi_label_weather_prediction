"""
    Author : Ye Yint Thu
    Email  : yeyintthu536@gmail.com
"""
from copy import deepcopy
import os
from sklearn.metrics import classification_report
import torch
import torchvision
from typing import List, Iterator, Tuple, Dict
from random import uniform


def criterion(loss_fn, preds, gts, device):
    losses = torch.zeros((len(preds.keys())), device=device)
    for i, key in enumerate(preds):
        losses[i] = loss_fn(preds[key], torch.unsqueeze(gts[key], 1).float().to(device))
    return torch.mean(losses)


def train(
    model: torch.nn.Module,
    device: torch.device,
    lr_rate: float,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    milestones: List[int],
    saved_dir: str,
) -> Tuple[torch.nn.Module, Iterator]:
    """Train the model with multi-steps learning rates for a given order of epochs
        and return the best checkpointed model, training and validation losses

    Args:
        model (torch.nn.Module): Nn module that represents a torch model
        device (torch.device): Device on which training process will be run
        lr_rate (float): Initial learning rate for training
        num_epochs (int): Number of epochs
        train_loader (torch.utils.data.DataLoader): Data loader for training set
        val_loader (torch.utils.data.DataLoader): Data loader for testing set
        Milestones (List[int]): Milestones of epochs for multi-steps learning process
        saved_dir (str): Directory to save model checkpoints

    Returns:
        Tuple[torch.nn.Modue, Iterator]: Training and validation losses through epochs
    """
    # define scheduled optimizer and loss func
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    loss_func = torch.nn.BCELoss()
    # initialize losses and weights
    train_epoch_loss = []
    validation_epoch_loss = []
    best_loss = 999.00
    model = model.to(device)
    best_model_wts = deepcopy(model.state_dict())
    # train and validate per epoch
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
            n_total_steps = len(data_loader["loader"])  # type: ignore
            for i, samples in enumerate(data_loader["loader"]):  # type: ignore
                images = samples["image_tensors"].to(device)
                gts = samples["labels"]
                scheduler.optimizer.zero_grad()  # type: ignore
                with torch.set_grad_enabled(data_loader["type"] == "train"):
                    preds = model(images)
                    loss = criterion(loss_func, preds, gts, device=device)
                    losses.append(loss.item())
                    if data_loader["type"] == "train":
                        loss.backward()
                        scheduler.optimizer.step()  # type: ignore
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
                            best_model_wts = deepcopy(model.state_dict())
                            print("Updating checkpoint..")
                            os.makedirs(saved_dir, exist_ok=True)
                            torch.save(
                                model.state_dict(),
                                os.path.join(saved_dir, "best_model.pt"),
                            )
                        if epoch == num_epochs - 1:
                            torch.save(
                                model.state_dict(),
                                os.path.join(saved_dir, "last_model.pt"),
                            )
                    if data_loader["type"] == "train":
                        train_epoch_loss.append(current_loss)
    model.load_state_dict(best_model_wts)
    return model, zip(*[train_epoch_loss, validation_epoch_loss])


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    classes: List,
) -> Dict:
    """Evaluate multi-label binary classifier and return classification reports
        for classifiers

    Args:
        model (torch.nn.Module): Nn module representation of multi-label
            binary classification
        device (torch.device): Device on which evalution will be run
        data_loader (_type_): Dataloader for evalution
        classes (List): List of classes

    Returns:
        Dict: Dict of classification reports for binary classifiers
    """
    model.eval()
    gts, preds = {}, {}  # type: ignore
    for class_ in classes:
        gts[class_] = []
        preds[class_] = []
    for i, samples in enumerate(data_loader):
        images = samples["image_tensors"].to(device)
        gts = samples["labels"]
        for class_, val in gts.items():
            gts[class_] = gts[class_] + [int(label) for label in val.tolist()]
        with torch.no_grad():
            preds = model(images)
        for class_, val in preds.items():
            preds[class_] = preds[class_] + [round(prob[0]) for prob in val.tolist()]

    classification_reports_dict = {
        class_: classification_report(
            gts[class_],
            preds[class_],
            target_names=[f"Not {class_}", class_],
        )
        for class_ in classes
    }
    return classification_reports_dict


def get_transforms(
    input_size: Tuple, mean: List, std: List, train_pipeline: bool = True
) -> torch.nn.Module:
    """Construct tarnsformation pipeline for train and validation/test operations

    Args:
        input_size (Tuple): Input size of model
        mean (List): Mean of the dataset
        std (List): Std of the dataset
        train_pipeline (bool, optional): Flag for training process. Defaults to True.

    Returns:
        torch.nn.Module: A stack of transformaton operations based on
            pipeline type (train/val)
    """
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )
    if train_pipeline:
        augmented_transform = torchvision.transforms.RandomChoice(
            [
                torchvision.transforms.RandomApply(
                    torch.nn.ModuleList([torchvision.transforms.RandomRotation(30)]),
                    p=0.4,
                ),
                torchvision.transforms.RandomApply(
                    torch.nn.ModuleList(
                        [
                            torchvision.transforms.CenterCrop(
                                size=int(round(uniform(0.1, 0.3), 2) * input_size[0])
                            )
                        ]
                    ),
                    p=0.4,
                ),
                torchvision.transforms.RandomHorizontalFlip(p=0.4),
            ]
        )
        transform_pipeline = torchvision.transforms.Compose(
            [augmented_transform, transforms]
        )

    else:
        transform_pipeline = transforms
    return transform_pipeline
