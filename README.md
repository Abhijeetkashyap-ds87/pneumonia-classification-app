# 🩺 Pneumonia Classification App

This is a deep learning–based web application that classifies **Chest X-ray images** as **Normal** or **Pneumonia** using a fine-tuned `MobileNetV3-Small` model. Built using **PyTorch**, **Streamlit**, and follows best practices in model deployment and reproducibility.

---

## 🚀 Features

- Upload chest X-ray images to get instant predictions.
- Clean and intuitive UI built with Streamlit.
- Inference-ready `.pth` model (no training code included).
- Compatible with local systems and cloud environments like **Kaggle Notebooks**.

---

## 🧠 Model Overview

- **Architecture**: MobileNetV3-Small (Pretrained on ImageNet, fine-tuned on pneumonia dataset)
- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `Adam`
- **Scheduler**: `ReduceLROnPlateau`
- **Evaluation Metrics**: Accuracy, Precision, Recall

---

## 🧰 Tech Stack

| Layer            | Tools/Frameworks Used                          |
|------------------|------------------------------------------------|
| Model Training   | PyTorch                                        |
| Model Saving     | `.pth` format                                  |
| Web Framework    | Streamlit                                      |
| Image Processing | PIL, Torchvision transforms                    |
| Visualization    | Matplotlib, Streamlit image components         |
| Deployment Ready | Requirements file + GitHub repo                |

---

---

## 💻 Installation & Usage

### 1. Clone the repository:
```bash
git clone https://github.com/Abhijeetkashyap-ds87/pneumonia-classification-app.git
cd pneumonia-classification-app
```
### Create a virtual environment
```bash
python3 -m venv kaggle_env
source kaggle_env/bin/activate
```
### Install required packages
```bash
pip install -r requirements.txt
```
### Run the app
```bash
streamlit run Homepage.py
```
