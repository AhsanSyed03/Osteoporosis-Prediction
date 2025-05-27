import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class OsteoporosisDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_map = {
            "normal": 0,
            "osteopenia": 1,
            "osteoporosis": 2
        }

        for label_name in self.class_map:
            class_dir = os.path.join(root_dir, label_name)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(class_dir, file))
                        self.labels.append(self.class_map[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class BMDDataset(Dataset):
    def __init__(self, csv_path, image_dir, bone_type_model, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.bone_type_model = bone_type_model.eval()
        self.device = next(bone_type_model.parameters()).device  # Get model's device

        data = pd.read_csv(csv_path)
        expected_cols = ['SPINE_BM', 'HIP_BMD', 'HIPNECK__HIPNECK_']
        existing_cols = [col for col in expected_cols if col in data.columns]

        self.samples = []
        self.skip_counters = {'missing_images': 0, 'missing_bmd_values': 0}

        for _, row in data.iterrows():
            identifier = str(row['IDENTIFIER_1'])

            if all(pd.isna(row.get(col)) for col in expected_cols):
                continue

            found_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                for root, _, files in os.walk(self.image_dir):
                    for file in files:
                        if identifier in file and file.lower().endswith(ext):
                            found_path = os.path.join(root, file)
                            break
                    if found_path:
                        break
                if found_path:
                    break

            if found_path:
                self.samples.append((found_path, row))
            else:
                self.skip_counters['missing_images'] += 1

        if not self.samples:
            raise ValueError("No valid samples in dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            image_path, row = self.samples[idx]
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = image

            image_tensor = image_tensor.to(self.device)  # Ensure tensor is on the correct device

            with torch.no_grad():
                bone_logits = self.bone_type_model(image_tensor.unsqueeze(0))
                bone_type_idx = bone_logits.argmax(1).item()
                bone_type = ["Spine", "Hip", "HipNeck"][bone_type_idx]

            if bone_type == "Spine":
                bmd = row['SPINE_BM']
            elif bone_type == "Hip":
                bmd = row['HIP_BMD']
            else:
                bmd = row['HIPNECK__HIPNECK_']

            if pd.isna(bmd):
                self.skip_counters['missing_bmd_values'] += 1
                raise RuntimeError(f"Missing BMD value for bone type {bone_type}")

            return image_tensor, torch.tensor(bmd, dtype=torch.float32)


        except Exception as e:
            self.skip_counters['missing_bmd_values'] += 1
            print(f"Skipping item {idx} due to error: {str(e)}")
            raise RuntimeError(f"Failed at index {idx}: {e}")
