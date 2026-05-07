import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)

    if model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(class_names))
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(class_names))
    else:
        raise ValueError("Model không hỗ trợ")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, img_size


@torch.no_grad()
def predict(model, image_path, class_names, img_size, device):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    outputs = model(x)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size = load_model(args.model_path, device)

    pred_class, confidence = predict(model, args.image_path, class_names, img_size, device)
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()

    main(args)