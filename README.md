# ♻️ Waste Classification and Recycling Management using Deep Learning

This project leverages Deep Learning and AI to automate the classification of waste from images and suggest eco-friendly recycling methods using a rule-based logic system. It addresses the pressing need for smart, efficient, and sustainable waste handling, ideal for deployment in smart cities, public infrastructure, and environmental conservation systems.

---

## 🎯 Objective

To build an intelligent, automated waste classification system that:
- Accepts waste images as input.
- Predicts the correct waste category using deep learning models.
- Recommends an appropriate recycling or disposal method using a rule-based system.
- Provides a deployable backend API for real-world applications.

---

## 📖 Problem Statement

Manual waste sorting is:
- Time-consuming and inconsistent.
- Prone to human error and inefficiency.
- A major contributor to low recycling success and increased landfills.

This system automates waste categorization, enhancing recycling rates, reducing contamination, and promoting environmental sustainability by:
- Improving classification accuracy using CNN-based models.
- Reducing manual effort and human intervention.
- Offering a scalable backend for integration with IoT, apps, or smart bins.

---

## 🚀 Proposed System

The pipeline of the proposed system includes:

1. 📤 **Input**: Upload an image (from user, camera, app, or smart bin).
2. 🧠 **Classification**: Use ResNet-50 or MobileNet to identify waste type.
3. ⚙️ **Recycling Decision**: Rule-based system provides optimal recycling instructions.
4. 📲 **Output**: Returns predicted class, confidence scores, and recycling method.

---

## 🧰 Tools and Technologies

| Category        | Technology Used                            |
|-----------------|---------------------------------------------|
| Language        | Python 3.x                                  |
| Libraries       | TensorFlow, Keras, NumPy, OpenCV, Scikit-learn |
| Models          | ResNet-50, MobileNet (Transfer Learning)    |
| Preprocessing   | Image resizing, augmentation, normalization |
| Deployment      | Flask (REST API), Postman (API testing)     |
| Development     | Kaggle Notebook (GPU)                       |
| Version Control | Git, GitHub                                 |

---

## 📂 Dataset Information

- **Dataset Name**: Garbage Classification v2  
- **Source**: [Kaggle - sumn2u](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data)
- **Total Images**: ~19,762 images
- **Image Format**: RGB (JPEG, PNG)
- **Classes (10)**:
  - 🧴 Plastic
  - 📦 Cardboard
  - 🥫 Metal
  - 📄 Paper
  - 🧪 Glass
  - 🗑 Trash
  - 🌱 Biological
  - 👟 Shoes
  - 👕 Clothes
  - 🔋 Battery

### 🧼 Preprocessing Steps:
- Resized to 224×224 (for CNNs)
- Augmentation: Rotation, flipping, zooming
- Normalization based on ImageNet mean & std

---

## 🛠️ Methodology

### 🔹 ResNet-50 (Model 1)
- Transfer Learning from ImageNet
- Frozen base, custom classifier head
- Optimized with Early Stopping & Learning Rate Scheduler
- Achieved:
  - **Train Accuracy**: 90.83%
  - **Test Accuracy**: 90.07%
  - **Best Class**: Clothes (F1: 0.96)

### 🔹 MobileNet (Model 2)
- Optimized for real-time and mobile use
- Lightweight with Depthwise Separable Convolutions
- Achieved:
  - **Train Accuracy**: 87.37%
  - **Best Class**: Shoes (F1: 0.92)

---

## 📊 Model Training & Evaluation

| Model     | Accuracy | Train Loss | Best F1 Score | Worst F1 Score |
|-----------|----------|------------|----------------|----------------|
| ResNet-50 | 90.83%   | 0.326      | Clothes (0.96) | Trash (0.79)   |
| MobileNet | 87.37%   | 0.366      | Shoes (0.92)   | Trash (~0.78)  |

📈 **Weighted Average F1 Score (ResNet)**: 0.90  
✅ Well-balanced classification across most categories.

---

## 🧠 Rule-Based Recycling System

After classification, the system applies **if-else decision logic** to determine the recycling method:

```python
rules = {
  "plastic": "Sorted, shredded, and sent to plastic recyclers",
  "glass": "Cleaned and melted for reuse",
  "trash": "Sent to landfill with proper disposal",
  "battery": "Sent to hazardous waste facilities",
  "paper": "Pulped and reused in paper mills",
  "clothes": "Donated or turned into rags",
  "metal": "Melted and remolded into new products",
  "shoes": "Reused or upcycled through NGOs",
  "cardboard": "Compressed and recycled into paper-based products",
  "biological": "Composted or sent to biomass plants"
}


## 🧠 Trained Keras Model

You can download the trained Keras model from the link below:

🔗 [Download Model from Google Drive](https://drive.google.com/file/d/1CxTDvCM1eVTttGqtHXRfDhHWtpQ34UYm/view?usp=sharing)
