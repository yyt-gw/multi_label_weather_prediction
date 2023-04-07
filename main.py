"""
    Author : Ye Yint Thu
    Email  : yeyintthu536@gmail.com
"""

import argparse
from enum import Enum
import mlflow
import os
import pickle
import yaml
import torch
from datetime import datetime
from typing import Iterator, List, Dict

# import custom packages
from dataset.multi_label_weather_dataset import MultiLabelWeatherDataset
from model.model import MultiLabelBinaryClassifier
from utils.utils import train, evaluate, get_transforms


# define dataset types
class OperationTypes(Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"
    VALIDATE = "validate"
    TRAIN_EVAL = "train_eval"


class MLFlowTracker:
    def __init__(
        self, exp_name: str, config: Dict, log_dir: str = "mlflow_logs"
    ) -> None:
        self.exp_name = exp_name
        self.config = config
        self.log_dir = log_dir
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment:
            self.experiment_id = experiment.experiment_id
        else:
            self.experiment_id = mlflow.create_experiment(exp_name)

    def init_run(self):
        mlflow.start_run(
            run_name=datetime.now().strftime("%H_%M_%S_%d_%M_%Y"),
            experiment_id=self.experiment_id,
        )

    def terminate_run(self):
        mlflow.end_run()

    def log_params(self):
        mlflow.log_params(
            {
                "Batch size": config["batch_size"],
                "Backbone type": config["backbone"]["type"],
                "Frozen backbone layers": config["backbone"]["frozen_layers"],
                "Input size": config["input_size"],
                "Learning rate": config["training"]["lr"],
                "Number of epochs": config["epochs"],
                "Milestones": config["milestones"],
            }
        )

    def log_config(self, config_path: str = "config.yml"):
        os.makedirs(self.log_dir, exist_ok=True)
        source_config_path = os.path.join(self.log_dir, config_path)
        with open(source_config_path, "w") as con_f:
            yaml.dump(config, con_f)
        mlflow.log_artifact(source_config_path, artifact_path=config_path)

    def log_losses(self, losses: Iterator):
        # start mlflow run
        for i, loss_info in enumerate(losses):
            mlflow.log_metrics(
                {"BCE train loss": loss_info[0], "BCE val Loss": loss_info[1]},
                i,
            )

    def log_model(self, model):
        mlflow.pytorch.log_model(
            model, artifact_path="best-model", pickle_module=pickle
        )


def get_dataloader(
    config: Dict, classes: List, transforms, for_train: bool = True
) -> torch.utils.data.DataLoader:
    """Get dataloader for the given process train/val

    Args:
        config (Dict): Config params
        classes (List): List of classes
        transforms (torch.nn.Module): Transformation pipeline
        for_train (bool, optional): Flag for training process. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Dataloader for the given process type
    """
    is_shuffle = True
    if for_train:
        dataset = MultiLabelWeatherDataset(
            config["dataset"]["train_path"],
            config["dataset"]["train_images_root"],
            transforms,
            classes,
        )
    else:
        dataset = MultiLabelWeatherDataset(
            config["dataset"]["val_path"],
            config["dataset"]["val_images_root"],
            transforms,
            classes,
        )
        is_shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=is_shuffle,
        num_workers=config["num_workers"],
    )


def operate_training(
    model: torch.nn.Module,
    config: Dict,
    device: torch.device,
    mlflow_tracker: MLFlowTracker,
):
    """Train the model with provided configuration params

    Args:
        model (torch.nn.Module): Nn module representation of multi-label
            binary classifier
        config (Dict): Dict of config params
        device (torch.device): Device on which training will be run
        mlflow_tracker (MLFlowTracker): Mlflow tracker for logging metrics,
            params and artifacts
    """
    # create transformation pipelines
    input_size = config["backbone"]["input_size"]
    mean = config["dataset"]["mean"]
    std = config["dataset"]["std"]

    train_transforms = get_transforms(input_size, mean, std)
    val_transforms = get_transforms(input_size, mean, std, train_pipeline=False)
    # create datasets and dataloaders for training and validation sets
    train_loader = get_dataloader(config, classes, train_transforms)
    val_loader = get_dataloader(config, classes, val_transforms, for_train=False)

    exp_ext = datetime.now().strftime("%H_%M_%S_%d_%M_%Y")
    weights_dir = os.path.join(*[config["saved_dir"], "weights", f"Exp_at_{exp_ext}"])

    # training
    best_model, losses = train(
        model,
        device,
        config["training"]["lr"],
        config["training"]["epochs"],
        train_loader,
        val_loader,
        config["training"]["milestones"],
        weights_dir,
    )
    # write training and validation losses to tensorboard
    mlflow_tracker.init_run()
    mlflow_tracker.log_losses(losses)
    mlflow_tracker.log_config()
    mlflow_tracker.log_params()
    mlflow_tracker.log_model(model=best_model)
    mlflow_tracker.terminate_run()


def operate_evaluation(
    model: torch.nn.Module, config: Dict, device: torch.device, classes: List
):
    # create transformation pipelines
    input_size = config["backbone"]["input_size"]
    mean = config["dataset"]["mean"]
    std = config["dataset"]["std"]
    val_transforms = get_transforms(input_size, mean, std, train_pipeline=False)
    # create datasets and dataloaders for training and validation sets
    val_loader = get_dataloader(config, classes, val_transforms, for_train=False)
    classification_reports_dict = evaluate(model, device, val_loader, classes)
    for _, classification_report in classification_reports_dict.items():
        print(classification_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Training and/or validation of multi-label binary classifier"
            + "for multi-labelweather prediction"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "operation_type",
        help=(
            "Use one of the follwing operations\n"
            + "  `train`     : run training operation\n"
            + "  `evaluate`  : run evaluation operation(generate metrics)\n"
            + "  `train_eval`: run training operation then evalution operation\n"
            + "  `validate`  : run validation operatoin(visualize Gts vs Preds)"
        ),
    )
    parser.add_argument(
        "--config_path",
        default="./config/config.yml",
        help="Config yml file for the training/evaluation/validation",
    )
    args = parser.parse_args()
    if not os.path.exists(args.config_path):
        raise FileNotFoundError("Config file is not found!")
    if not args.config_path.endswith(".yml"):
        raise NotImplementedError(
            "Unsupported config file type! Please use yml config file!"
        )

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
    with open(config["dataset"]["classes_path"], "r") as f:
        classes = f.read().rstrip().split("\n")
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    # create classifier
    classifier = MultiLabelBinaryClassifier(
        classes, config["backbone"]["type"], config["backbone"]["frozen_layers"], device
    )
    if args.operation_type == OperationTypes.TRAIN.value:
        mlflow_tracker = MLFlowTracker("test-exp", config)
        operate_training(classifier, config, device, mlflow_tracker)
    elif args.operation_type == OperationTypes.EVALUATE.value:
        operate_evaluation(classifier, config, device, classes)
    elif args.operation_type == OperationTypes.TRAIN_EVAL.value:
        mlflow_tracker = MLFlowTracker("test-exp", config)
        operate_training(classifier, config, device, mlflow_tracker)
        operate_evaluation(classifier, config, device, classes)
