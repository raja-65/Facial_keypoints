# Facial Keypoints Detection with CNN  
This repository contains a **Facial Keypoints Detection** pipeline using a **Convolutional Neural Network (CNN)** implemented in PyTorch. The project processes facial images, detects 30 facial keypoints, and prepares a submission for Kaggle’s **Facial Keypoints Detection** challenge.

---

## 📂 Dataset Setup  
Before running the scripts, **download and extract** the dataset from Kaggle:

🔗 **Kaggle Dataset**: [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/data)

### 🔽 **Steps to Download & Extract**
1. Go to the [Kaggle dataset link](https://www.kaggle.com/c/facial-keypoints-detection/data).
2. Download **ALL ZIP files** (including subfolders).
3. Extract all contents into the **root directory** of this project.


## 🚀 Project Overview  
This project trains a CNN model to detect key facial landmarks such as eyes, eyebrows, nose, and mouth positions from grayscale images of faces.  

**Pipeline Steps:**  
1. **Data Preprocessing** – Convert images from CSV format to a structured NumPy array.  
2. **Data Augmentation** – Apply flipping to enhance model generalization.  
3. **Model Training** – Train a deep CNN with masked loss for missing keypoints.  
4. **Prediction & Submission** – Predict keypoints on the test set and generate a CSV submission.  

---

## 📂 File Structure  
```
|-- facial-keypoints-detection  
    |-- training.csv          # Training data with keypoints and images  
    |-- test.csv              # Test data with images only  
    |-- IdLookupTable.csv      # Mapping for submission  
    |-- SampleSubmission.csv   # Example submission format  
    |-- facial_keypoints.py    # CNN Model definition  & training
    |-- onlytesting.py         # Testing & prediction script  
    |-- submission.csv         # Final submission file  
    |-- README.md              # Project documentation  
```

---

## ⚙️ Installation  
Ensure you have Python **3.7+** installed. Then, install the required dependencies:

```bash
remember to pip install all requirements
```
  
If using **GPU**, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Dataset  
- The dataset consists of **96x96 grayscale facial images**. on kaggle  has the DB 
- Each image has **up to 30 keypoints (x, y coordinates)**.  
- Some keypoints may be missing (handled using a mask).  
- The dataset is provided as **CSV files**, where images are stored as pixel intensity values.  

---

## 🔧 Model Architecture  
The model is a **CNN-based regression network** with the following structure:  

| Layer           | Output Shape | Activation |
|----------------|-------------|------------|
| Conv2D(32, 3x3) | 96x96x32    | ReLU       |
| MaxPool2D(2x2)  | 48x48x32    | -          |
| Conv2D(64, 3x3) | 48x48x64    | ReLU       |
| MaxPool2D(2x2)  | 24x24x64    | -          |
| Conv2D(128, 3x3) | 24x24x128  | ReLU       |
| MaxPool2D(2x2)  | 12x12x128   | -          |
| Flatten        | 18432        | -          |
| Dense(256)     | 256          | ReLU       |
| Dropout(0.5)   | -           | -          |
| Dense(30)      | 30 keypoints | -          |

---

## 🏋️‍♂️ Training the Model  
## specially trained on CPU
Run the training script:  
```bash
python facial_keypoints.py
```
The model will be saved as:  
```bash
facial_keypoints_model.pth
```

### 🏆 Training Performance
- Optimizer: **Adam** (learning rate = 0.001)  
- Loss function: **Masked Mean Squared Error (MSE)**  
- Trained for **50 epochs** with **batch size = 32**  

---

## 🔎 Making Predictions  
Run the prediction and testing script:  
```bash
python onlytesting.py
```
This will generate a `submission.csv` file.

### 📄 Submission Format  
The output file should be in the following format:  
```
RowId,Location
1,66.033563
2,39.002274
3,62.538623
...
```
Where **Location** is the predicted coordinate for a given **RowId**.

---

## 🖥️ Example Usage  
### **Training Output**  
```
Epoch 1/50, Train Loss: 0.0054, Val Loss: 0.0048, Val RMSE: 0.0692
Epoch 2/50, Train Loss: 0.0041, Val Loss: 0.0037, Val RMSE: 0.0608
...
```
  
### **Prediction Output**  
```
Making predictions on the test data...
Generating submission file...
Submission saved as 'submission.csv'
```

---

## 🛠 Troubleshooting  
| Issue | Solution |
|--------|------------|
| **FileNotFoundError: training.csv not found** | Ensure all dataset files are in the working directory. |
| **CUDA out of memory** | Reduce `batch_size` in `train.py`. |
| **Predictions are all zeros** | Ensure the model is correctly trained and loaded. |
| **Mismatch in feature names** | Ensure `IdLookupTable.csv` matches the correct keypoint order. |

---

## 📌 Key Takeaways  
✅ CNNs work well for **keypoint regression**.  
✅ **Data augmentation** (horizontal flipping) improves performance.  
✅ **Masked loss** helps handle missing values.  
✅ Proper **preprocessing and normalization** are crucial.  

---

## 📜 License  
This project is **MIT Licensed**. You are free to use, modify, and distribute the code.

---

## 🙌 Acknowledgments  
- Kaggle's **Facial Keypoints Detection** competition  
- PyTorch documentation for deep learning models
- 
---

🚀 **Happy Coding!** 🚀
