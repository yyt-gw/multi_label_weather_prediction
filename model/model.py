import torch
import torchvision


def freeze_layers(model, freeze_count):
    """
    Freeze the weights of pretrained model
    """
    layer_count = 0
    for child in model.children():
        layer_count += 1
        if layer_count <= freeze_count:
            for param in child.parameters():
                param.requires_grad = False
    print(f"Number of layer : {layer_count}")
    return model


class MultiLabelBinaryClassifier(torch.nn.Module):
    def __init__(self, classes_, frozen_layers=2, device="cuda"):
        super().__init__()
        self.resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        self.resnet = freeze_layers(self.resnet, frozen_layers)
        self.backbone = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.binary_classifier_heads = {}
        for i, class_ in enumerate(classes_):
            self.binary_classifier_heads[class_] = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(in_features=2048, out_features=128),
                torch.nn.Dropout(p=0.2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=128, out_features=1),
                torch.nn.Sigmoid(),
            ).to(device)

    def forward(self, x):
        features = torch.flatten(self.backbone(x), 1)
        out = {}
        for key, classifier_val in self.binary_classifier_heads.items():
            out[key] = classifier_val(features)
        return out
