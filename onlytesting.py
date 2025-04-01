# Import libraries
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


# Define the CNN model (must be the same architecture as the trained model)
class FacialKeypointsCNN(nn.Module):
    def __init__(self):
        super(FacialKeypointsCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x96x96
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 32x48x48
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x48x48
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 64x24x24
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128x24x24
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: 128x12x12  <-- Changed here
        self.flatten = nn.Flatten()  # Output: 128*12*12 = 18432
        self.fc1 = nn.Linear(18432, 256)  # <-- Changed input size
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


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for testing: {device}")

# Load the trained model
print("Loading the exported model...")
model = FacialKeypointsCNN().to(device)  # Instantiate the model
try:
    model.load_state_dict(torch.load("facial_keypoints_model.pth", map_location=device))
except FileNotFoundError:
    print(
        "Error: 'facial_keypoints_model.pth' not found. Make sure the file is in the same directory."
    )
    exit()
model.eval()  # Set the model to evaluation mode

# Load test data
print("Loading test data...")
try:
    test_df = pd.read_csv("test.csv")
except FileNotFoundError:
    print("Error: 'test.csv' not found. Make sure the file is in the same directory.")
    exit()

# Check if "Image" column exists
if "ImageId" not in test_df.columns or "Image" not in test_df.columns:
    raise KeyError("Error: 'ImageId' or 'Image' column not found in test_df.")

# Preprocess test images
test_images = []
image_ids = test_df["ImageId"].tolist()
print("Processing test data...")
for i in range(len(test_df)):
    img_str = test_df["Image"][i]
    img = np.array([float(p) for p in img_str.split(" ")]).reshape(96, 96, 1) / 255.0
    test_images.append(img)

test_images = np.array(test_images)  # Shape: (num_samples, 96, 96, 1)


# Define test dataset
class TestDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return torch.from_numpy(image).permute(2, 0, 1).float()  # Shape: (1, 96, 96)


test_dataset = TestDataset(test_images)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Make predictions on the test data
print("Making predictions on the test data...")
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        images = batch.to(device)
        preds = model(images).cpu().numpy()
        all_preds.append(preds)

# Concatenate predictions
test_predictions = np.vstack(all_preds)  # Shape: (num_test_images, 30)

# Load the IdLookupTable
print("Loading IdLookupTable...")
try:
    id_lookup_df = pd.read_csv("IdLookupTable.csv")
except FileNotFoundError:
    print(
        "Error: 'IdLookupTable.csv' not found. Make sure the file is in the same directory."
    )
    exit()

# Prepare submission DataFrame
print("Preparing submission file...")
submission_df = pd.DataFrame(columns=["RowId", "Location"])

# Define the order of keypoint names as predicted by your model
feature_names = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "left_eye_inner_corner_x",
    "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",
    "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",
    "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",
    "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",
    "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",
    "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",
    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",
    "right_eyebrow_outer_end_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y",
    "mouth_center_top_lip_x",
    "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
]

# Fill the submission DataFrame based on IdLookupTable
for index, row in id_lookup_df.iterrows():
    row_id = row["RowId"]
    image_id = row["ImageId"]
    feature_name = row["FeatureName"]

    try:
        image_index = image_ids.index(image_id)
    except ValueError:
        print(f"Warning: ImageId {image_id} not found in test predictions.")
        continue

    try:
        feature_index = feature_names.index(feature_name)
        location = test_predictions[image_index, feature_index]
        submission_df = pd.concat(
            [submission_df, pd.DataFrame([{"RowId": row_id, "Location": location}])],
            ignore_index=True,
        )
    except ValueError:
        print(f"Warning: FeatureName {feature_name} not in the defined feature list.")

# Save submission file
submission_df.to_csv("submission.csv", index=False)

print("Testing complete! Submission file saved as 'submission.csv'")
