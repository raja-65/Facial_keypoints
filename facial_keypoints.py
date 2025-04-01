# Import libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Load the data
print("Loading data...")
train_df = pd.read_csv("training.csv")
test_df = pd.read_csv("test.csv")
submission_df = pd.read_csv("SampleSubmission.csv")

# Preprocess training data
images = []
targets = []
masks = []

print("Processing training data...")
for i in range(len(train_df)):
    # Convert image string to 96x96x1 array and normalize to [0, 1]
    img_str = train_df["Image"][i]
    img = np.array([float(p) for p in img_str.split(" ")]).reshape(96, 96, 1) / 255.0
    images.append(img)

    # Extract the 30 coordinates, handle missing values with a mask
    target = train_df.iloc[i, :30].values.astype(float)
    mask = ~np.isnan(target)  # 1 where coordinate exists, 0 where NaN
    target[np.isnan(target)] = 0.0  # Replace NaN with 0 for PyTorch
    targets.append(target)
    masks.append(mask)

images = np.array(images)  # Shape: (7049, 96, 96, 1)
targets = np.array(targets)  # Shape: (7049, 30)
masks = np.array(masks).astype(float)  # Shape: (7049, 30)

# Split into training (80%) and validation (20%) sets
train_images, val_images, train_targets, val_targets, train_masks, val_masks = (
    train_test_split(images, targets, masks, test_size=0.2, random_state=42)
)


# Define a custom dataset with optional data augmentation
class FacialKeypointsDataset(Dataset):
    def __init__(self, images, targets, masks, augment=False):
        self.images = images
        self.targets = targets
        self.masks = masks
        self.augment = augment
        # Permutation to swap left and right key points for flipping
        if augment:
            self.perm = np.array(
                [
                    2,
                    3,
                    0,
                    1,
                    8,
                    9,
                    10,
                    11,
                    4,
                    5,
                    6,
                    7,
                    16,
                    17,
                    18,
                    19,
                    12,
                    13,
                    14,
                    15,
                    20,
                    21,
                    24,
                    25,
                    22,
                    23,
                    26,
                    27,
                    28,
                    29,
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        mask = self.masks[idx]

        # Randomly flip image and adjust coordinates (50% chance)
        if self.augment and np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()  # Flip horizontally
            target_swapped = target[self.perm]  # Swap left/right key points
            mask_swapped = mask[self.perm]
            even_indices = np.arange(0, 30, 2)  # x coordinates
            target_swapped[even_indices] = (
                95 - target_swapped[even_indices]
            )  # Mirror x: 95 - x
            target = target_swapped
            mask = mask_swapped

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Shape: (1, 96, 96)
        target = torch.from_numpy(target).float()
        mask = torch.from_numpy(mask).float()
        return image, target, mask


# Create datasets
train_dataset = FacialKeypointsDataset(
    train_images, train_targets, train_masks, augment=True
)
val_dataset = FacialKeypointsDataset(val_images, val_targets, val_masks, augment=False)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define the CNN model
class FacialKeypointsCNN(nn.Module):
    def __init__(self):
        super(FacialKeypointsCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x96x96
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 32x48x48
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x48x48
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 64x24x24
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128x24x24
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: 128x12x12
        self.flatten = nn.Flatten()  # Output: 128*12*12 = 18432
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 30)  # Output: 30 coordinates

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Define the masked MSE loss
def masked_mse_loss(pred, target, mask):
    squared_errors = (pred - target) ** 2
    masked_errors = squared_errors * mask
    total_loss = masked_errors.sum()
    num_valid = mask.sum()
    return total_loss / num_valid  # Average MSE over valid coordinates


# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and optimizer
model = FacialKeypointsCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, targets, masks in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = masked_mse_loss(preds, targets, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_rmse = 0.0
    with torch.no_grad():
        for images, targets, masks in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = masked_mse_loss(preds, targets, masks)
            val_loss += loss.item() * images.size(0)

            # Compute RMSE
            squared_errors = (preds - targets) ** 2
            masked_errors = squared_errors * masks
            mse = masked_errors.sum() / masks.sum()
            rmse = torch.sqrt(mse)
            val_rmse += rmse.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_rmse /= len(val_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}"
    )

# Export the model
print("Exporting model...")
torch.save(model.state_dict(), "facial_keypoints_model.pth")
print("Model exported successfully.")
