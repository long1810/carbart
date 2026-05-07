import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


def build_model(model_name, num_classes):
    if model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Model không hỗ trợ")
    return model


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint["model_name"]
    img_size = checkpoint.get("img_size", 224)

    model = build_model(model_name, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, img_size


@torch.no_grad()
def predict(model, image, class_names, img_size, device):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    x = transform(image).unsqueeze(0).to(device)
    outputs = model(x)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)
    return class_names[pred.item()], conf.item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cabbage_model, cabbage_classes, cabbage_size = load_model(args.cabbage_model, device)
    pest_model, pest_classes, pest_size = load_model(args.pest_model, device)

    image = Image.open(args.image_path).convert("RGB")

    cabbage_pred, cabbage_conf = predict(cabbage_model, image, cabbage_classes, cabbage_size, device)
    print(f"[Bước 1] Cabbage classifier: {cabbage_pred} ({cabbage_conf:.4f})")

    if cabbage_pred != "cabbage":
        print("Kết luận cuối: Không phải cây bắp cải -> dừng")
        return

    pest_pred, pest_conf = predict(pest_model, image, pest_classes, pest_size, device)
    print(f"[Bước 2] Pest classifier: {pest_pred} ({pest_conf:.4f})")

    if pest_pred == "pest":
        print("Kết luận cuối: Cây bắp cải có sâu")
    else:
        print("Kết luận cuối: Cây bắp cải không có sâu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--cabbage_model", type=str, required=True)
    parser.add_argument("--pest_model", type=str, required=True)
    args = parser.parse_args()

    main(args)