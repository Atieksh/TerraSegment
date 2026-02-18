print("🚀 FAST TRAINING STARTED")

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

DEVICE = "cpu"
NUM_CLASSES = 10
BATCH_SIZE = 2
EPOCHS = 2
LR = 1e-4

BASE_PATH = r"C:\Users\Atieksh\Documents\h"

TRAIN_IMG = os.path.join(BASE_PATH, "Offroad_Segmentation_Training_Dataset",
                         "Offroad_Segmentation_Training_Dataset",
                         "train", "Color_Images")

TRAIN_MASK = os.path.join(BASE_PATH, "Offroad_Segmentation_Training_Dataset",
                          "Offroad_Segmentation_Training_Dataset",
                          "train", "Segmentation")

class DatasetSeg(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # 🔥 LIMIT DATA FOR SPEED
        self.images = os.listdir(img_dir)[:50]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            return self.__getitem__((idx + 1) % len(self))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mapping = {100:0,200:1,300:2,500:3,550:4,600:5,700:6,800:7,7100:8,10000:9}
        new_mask = np.zeros_like(mask)

        for k, v in mapping.items():
            new_mask[mask == k] = v

        mask = new_mask

        img = cv2.resize(img, (256,256))
        mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)

        img = img / 255.0

        img = torch.tensor(img).permute(2,0,1).float()
        mask = torch.tensor(mask).long()

        return img, mask

def get_model():
    return smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)

def train():
    dataset = DatasetSeg(TRAIN_IMG, TRAIN_MASK)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        print(f"\n🔥 Epoch {epoch+1} started")

        total_loss = 0
        model.train()

        for i, (img, mask) in enumerate(loader):
            print(f"➡️ Batch {i+1}")

            img, mask = img.to(DEVICE), mask.to(DEVICE)

            out = model(img)
            loss = loss_fn(out, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} Loss: {total_loss:.4f}")

    # 🔥 FORCE SAVE
    save_path = r"C:\Users\Atieksh\Documents\h\model.pth"
    torch.save(model.state_dict(), save_path)

    print("\n🎉 MODEL SAVED SUCCESSFULLY")
    print(f"📁 Location: {save_path}")

if __name__ == "__main__":
    train()
