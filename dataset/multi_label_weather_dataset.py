import torch
from torchvision import transforms
from PIL import Image
import os


class MultiLabelWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, image_root, sourceTransform, weather_classes):
        self.image_root = image_root
        with open(data_file, "r") as f:
            self.data = [sample.replace("\n", "") for sample in f.readlines()]
        self.transform = sourceTransform
        self.weather_classes = weather_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_info = self.data[idx].split()
        filename, labels = sample_info[0], sample_info[1:]
        img = Image.open(os.path.join(self.image_root, filename)).convert("RGB")

        if self.transform:
            img = self.transform(img)
        labels = [float(label) for label in labels]
        data_info = {"image_tensors": img, "labels": {}}
        for class_, val in zip(*[self.weather_classes, labels]):
            data_info["labels"][class_] = val
        return data_info


if __name__ == "__main__":
    transforms = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    with open("../data/classes.txt", "r") as f:
        weather_classes = [class_.replace("\n", "") for class_ in f.readlines()]

    dataset = MultiLabelWeatherDataset(
        "../data/test.txt", "../data/test/", transforms, weather_classes
    )
    data_loader = torch.utils.data.DataLoader(  # type: ignore
        dataset, batch_size=4, shuffle=False, num_workers=4
    )

    for batch_info in data_loader:
        print(f"Image size : {batch_info['image_tensors'].size()}")
        print(f"Labels : {batch_info['labels']}")
        break
