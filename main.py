import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import time
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Compute Device Used is", device)

# Mappings
bone_type_map = {0: "Knee", 1: "Spine", 2: "Hip"}
condition_map = {0: "normal", 1: "osteopenia", 2: "osteoporosis"}
condition_description = {
    "normal": "Normal bone density, healthy bones.",
    "osteopenia": "Lower than normal bone density, increased risk for osteoporosis.",
    "osteoporosis": "Significantly lower bone density, high risk of fractures."
}

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load bone type model
bone_model = models.vgg19(weights=None)
bone_model.classifier[6] = nn.Linear(bone_model.classifier[6].in_features, 3)
bone_model = bone_model.to(device).eval()

try:
    bone_model.load_state_dict(
        torch.load("models/bone_type_model_vgg19.pth",weights_only=True, map_location=device),
        strict=True
    )
    print("Bone type model loaded successfully.")
except Exception as e:
    print("Failed to load bone type model:", e)
    exit()


# Load osteoporosis condition model
condition_base_model = models.resnet50(weights=None)
condition_base_model.fc = nn.Linear(condition_base_model.fc.in_features, 3)
condition_model = nn.Sequential(condition_base_model, nn.Dropout(0.3)).to(device).eval()
try:
    condition_model.load_state_dict(
        torch.load("models/osteoporosis_model_resnet50.pth", weights_only=True, map_location=device),
        strict=True
    )
    print("Osteoporosis model loaded successfully.")
except Exception as e:
    print("Failed to load condition model:", e)
    exit()

# Accuracy loading
def load_accuracy(filepath):
    try:
        with open(filepath, "r") as f:
            return float(f.read().strip())
    except:
        return None

bone_accuracy = load_accuracy("logs/bonetype_accuracy_vgg19.txt")
condition_accuracy = load_accuracy("logs/osteoporosis_accuracy_resnet50.txt")

# Deterministic BMD estimation
def estimate_bmd_from_condition(cond_class):
    if cond_class == "normal":
        return 1.20
    elif cond_class == "osteopenia":
        return 0.90
    elif cond_class == "osteoporosis":
        return 0.65
    return None

# Prediction function for UI and import
def run_prediction(image_path):
    try:
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            bone_output = bone_model(image)
            bone_probs = nn.Softmax(dim=1)(bone_output)
            bone_conf, bone_pred = torch.max(bone_probs, 1)
            bone_type = bone_type_map[bone_pred.item()]

            condition_output = condition_model(image)
            condition_probs = nn.Softmax(dim=1)(condition_output)
            cond_conf, cond_pred = torch.max(condition_probs, 1)
            cond_class = condition_map[cond_pred.item()]
            description = condition_description[cond_class]
            confidence = min(cond_conf.item() * 100, 99.0)
            estimated_bmd = estimate_bmd_from_condition(cond_class)

        end_time = time.time()
        inference_time = end_time - start_time

        return {
            "bone_type": bone_type,
            "condition": cond_class,
            "description": description,
            "confidence": confidence,
            "inference_time": inference_time,
            "estimated_bmd": estimated_bmd,
            "bone_accuracy": bone_accuracy,
            "condition_accuracy": condition_accuracy,
        }

    except Exception as e:
        print("Prediction error:", e)
        return None

# CLI usage
if __name__ == "__main__":
    def main():
        image_path = input("Enter the path of the image to predict: ").strip().strip('"')
        if not os.path.exists(image_path):
            print(f"Error: The image at {image_path} was not found.")
            return

        results = run_prediction(image_path)
        if results:
            print(f"\nSelected Image Path: {image_path}")
            print(f"Bone Type: {results['bone_type']}")
            print(f"Predicted Class: {results['condition']}")
            print(f"Description: {results['description']}")
            print(f"Confidence Score: {results['confidence']:.2f}%")
            print(f"Inference Time: {results['inference_time']:.4f} seconds")
            print(f"Estimated BMD: {results['estimated_bmd']} g/cm²")

            if results['bone_accuracy'] is not None:
                print(f"Bone Type Model Accuracy: {results['bone_accuracy']:.2f}%")
            else:
                print("Bone Type Model Accuracy: Not available")

            if results['condition_accuracy'] is not None:
                print(f"Osteoporosis Model Accuracy: {results['condition_accuracy']:.2f}%")
            else:
                print("Osteoporosis Model Accuracy: Not available")

    main()