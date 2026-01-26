# 🌿 Plant Disease Classifier using CNN

A deep learning project that detects plant diseases from leaf images using Convolutional Neural Networks (CNN).

---


---

## 📋 Project Overview

This project uses a custom CNN model trained on the **New Plant Diseases Dataset** to classify plant leaf images into different disease categories. The model achieves high accuracy in identifying various plant diseases, making it useful for farmers and agricultural professionals.

## 🚀 Deploy to Appwrite Functions (GitHub Connected)

1. Ensure the repo contains `app.py`, `requirements.txt`, `models/plant_disease_model.keras`, and `models/class_names.json`.
2. Edit `appwrite.json` and set `projectId` to your Appwrite project ID.
3. In Appwrite Console, create a Function from this repository/branch:
    - Runtime: Python 3.10
    - Path/workdir: repo root (matches `path: "."` in appwrite.json)
    - Entrypoint: `app.main`
    - Build command: `pip install -r requirements.txt`
    - Memory/timeout: start with 2048 MB and 30s to handle TensorFlow load.
4. Deploy; Appwrite installs deps and bundles the `models/` folder.
5. Invoke with raw bytes: `curl -X POST -H "Content-Type: image/jpeg" --data-binary @leaf.jpg https://<function-endpoint>`.

## 🎯 Key Features

- **Custom CNN Architecture**: Built from scratch without transfer learning
- **Web Interface**: Simple Streamlit app for easy image upload and prediction
- **Multi-class Classification**: Identifies 38 different plant disease classes
- **Confidence Scores**: Displays prediction confidence for transparency
- **Predictions**: it Shows most likely disease

## 📊 Dataset

- **Dataset Name**: New Plant Diseases Dataset (Augmented)
- **Source**: [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes**: 38 different plant disease categories
- **Total Images**: 87,000+ images

The dataset includes common diseases affecting plants like:
- Tomato diseases (Late Blight, Early Blight, etc.)
- Potato diseases
- Pepper diseases
- And many more

## 📂 Project Structure

```
plant-disease-classifier/
│
├── models/ # Saved model folder
│ ├── plant_disease_model.keras # Trained model file
│ └── class_names.json # List of disease classes
│
├── notebooks/
│ └── train_model.ipynb # Google Colab training notebook
│
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore file
```
## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web app framework
- **Python**: Programming language
- **NumPy**: Numerical computations
- **PIL**: Image processing
- **Google Colab**: Training environment (free T4 GPU)

## 📊 Model Architecture

The CNN model consists of:
- 5 Convolutional blocks (32, 64, 128, 256, 512 filters)
- Batch Normalization layers
- MaxPooling layers
- Dropout layers (0.5 and 0.3)
- Dense layers (512, 256 neurons)
- Output layer with softmax activation

**Input**: 128x128 RGB images
**Output**: 38 disease classes

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Kaggle account (for dataset download)
- Google Colab account (for training)

## 📈 Model Performance

- **Validation Accuracy**: **88.29%**
- **Number of Classes**: 38
- **Training Time**: ~50-60 minutes for **10** EPOCHS in Google Colab T4 GPU

## 📚 What I Learned
- **Deep Learning Basics** :Learned how CNNs extract image features using layers like convolution, pooling, and dense blocks.
- **Data Prep & Augmentation** : Understood how to clean, split, and augment image datasets for better model performance.
- **Cloud GPU Training** : Practiced training models on Google Colab using T4 GPUs and managing datasets through the Kaggle API.

## 🎓 Future Improvements
- Try transfer learning (ResNet, MobileNet)
- Add more plant species
- Improve UI/UX

## 🙏 Acknowledgments

- Dataset provided by Kaggle
- TensorFlow and Streamlit communities
- Google Colab for free GPU access

---