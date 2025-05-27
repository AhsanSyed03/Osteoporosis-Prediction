# Osteoporosis Detection Using Deep Learning

This project uses deep learning to classify X-ray and DXA images of knee and spine into three categories:
- Normal
- Osteopenia
- Osteoporosis

Built using PyTorch, the model is trained to support medical image-based classification to assist in early detection of osteoporosis.

---

## 📁 Project Structure

Osteo_IV_prj/ 
├── datasets/ 
│ ├── knee/ 
│ │ ├── normal/ 
│ │ ├── osteopenia/ 
│ │ └── osteoporosis/ 
│ └── spine/ 
│ ├── normal/ 
│ ├── osteopenia/ 
│ └── osteoporosis/ 
├── models/ 
│ └── osteoporosis_model.pth 
├── utils/ 
│ └── dataset.py 
│ └── transforms.py 
├── train.py 
├── main.py 
├── requirements.txt 
└── README.md


---

## 🚀 Getting Started

### 📦 Dataset

The dataset is organized into two folders: `knee` and `spine`. Each folder contains subfolders for normal, osteopenia, and osteoporosis images. The images are stored in PNG,JPG,JPEG format.

### 🛠️ Installation

To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```
### 🏃‍♀️ Training

To train the model, run the following command:
```bash
python train.py
```
This will train the model and save the trained model to `models/osteoporosis_model.pth`.

### 🧪 Testing

To test the model, run the following command:
```bash
python main.py
```
This will test the model and output the results to the console.

### 📖 Documentation

For more information on the project, refer to the ChatGPT file.

### 📝 License

This project is licensed under the MIT License.
---

