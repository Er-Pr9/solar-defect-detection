from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

GRADCAM_AVAILABLE = False
GRADCAM_IMPORT_ERROR = None

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    GRADCAM_AVAILABLE = True
except Exception as e:
    GRADCAM_IMPORT_ERROR = str(e)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_inference_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_path: str | Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Place the .pth checkpoint in the same folder as app.py or update MODEL_PATH in config.py."
        )

    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)

    if "class_names" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint must contain 'class_names' and 'model_state_dict'.")

    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def prepare_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")
    return image.convert("RGB")


def predict_topk(
    image: Image.Image,
    model: nn.Module,
    transform: transforms.Compose,
    class_names: List[str],
    device: torch.device,
    k: int = 3,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    image = prepare_image(image)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(k, len(class_names)))
    top_results = [(class_names[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_idxs)]

    predicted_class, confidence = top_results[0]
    return predicted_class, confidence, top_results


def predict_image(
    image: Image.Image,
    model: nn.Module,
    transform: transforms.Compose,
    class_names: List[str],
    device: torch.device,
) -> Tuple[str, float]:
    predicted_class, confidence, _ = predict_topk(image, model, transform, class_names, device, k=3)
    return predicted_class, confidence


def generate_gradcam(
    image: Image.Image,
    model: nn.Module,
    transform: transforms.Compose,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, object]:
    if not GRADCAM_AVAILABLE:
        error_msg = GRADCAM_IMPORT_ERROR or "Unknown Grad-CAM import error."
        raise ImportError(
            f"Grad-CAM could not be imported. Original error: {error_msg}"
        )

    image = prepare_image(image).resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.array(image).astype(np.float32) / 255.0
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    pred_class_idx = pred_idx.item()
    pred_class_name = class_names[pred_class_idx]
    confidence_score = confidence.item()

    target_layers = [model.layer4[-1]]
    targets = [ClassifierOutputTarget(pred_class_idx)]

    cam = None
    try:
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    finally:
        if cam is not None:
            cam.clear_hooks()

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return {
        "predicted_class": pred_class_name,
        "confidence": confidence_score,
        "original_image": (rgb_img * 255).astype(np.uint8),
        "gradcam_overlay": visualization,
        "heatmap": grayscale_cam,
    }