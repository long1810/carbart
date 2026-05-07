import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
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
def predict_patch(model, patch_bgr, class_names, img_size, device):
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    patch_pil = Image.fromarray(patch_rgb)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    x = transform(patch_pil).unsqueeze(0).to(device)
    outputs = model(x)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs


def sliding_window_inference(image, model, class_names, img_size, device,
                             patch_size=128, stride=64, threshold=0.8):
    h, w = image.shape[:2]
    boxes = []

    pest_index = class_names.index("pest") if "pest" in class_names else None
    if pest_index is None:
        raise ValueError("Class 'pest' không tồn tại trong model")

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            pred_class, conf, probs = predict_patch(model, patch, class_names, img_size, device)

            pest_prob = probs[pest_index]
            if pest_prob >= threshold:
                boxes.append((x, y, x + patch_size, y + patch_size, pest_prob))

    return boxes


def non_max_suppression(boxes, iou_threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    scores = boxes_np[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes[i])

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def draw_boxes(image, boxes):
    output = image.copy()
    for (x1, y1, x2, y2, score) in boxes:
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(output, f"pest:{score:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return output


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size = load_model(args.model_path, device)

    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {args.image_path}")

    boxes = sliding_window_inference(
        image=image,
        model=model,
        class_names=class_names,
        img_size=img_size,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        threshold=args.threshold
    )

    boxes = non_max_suppression(boxes, iou_threshold=0.3)
    output = draw_boxes(image, boxes)

    cv2.imwrite(args.output_path, output)
    print(f"Số vùng nghi có sâu: {len(boxes)}")
    print(f"Đã lưu kết quả tại: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output.jpg")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    main(args)