# 🫁 PneumoVision AI - Chest X-Ray Pneumonia Detection Web Application
### Powered by VGG16 + VGG19 Dual-Branch Hybrid Deep Learning & Explainable AI (Grad-CAM)

This website is a complete medical-grade diagnostic decision support web application built with **Flask**, **TensorFlow/Keras**, and **Vanilla HTML/CSS/JS**. It analyzes chest radiographs to detect **Pneumonia** vs **Normal** cases, displays **Grad-CAM** attention heatmaps, and provides evidence-based clinical recommendations.

---

## 📁 Project Structure

```
Chest_xray website/
├── App.py                                   # Flask backend server with Hybrid model & Grad-CAM engine
├── requirements.txt                         # Python dependencies
├── run.bat                                  # 1-Click launcher script for Windows
├── README.md                                # Setup & usage documentation
├── static/
│   ├── css/
│   │   └── style.css                        # Modern medical dark UI design system (responsive)
│   ├── js/
│   │   └── main.js                          # Drag-and-drop upload, live preview, print report
│   ├── samples/                             # Built-in sample X-ray images
│   └── uploads/                             # Uploaded X-rays and generated Grad-CAM heatmaps
└── templates/
    ├── index.html                           # Home / Diagnostic workspace with 1-click samples
    ├── result.html                          # Diagnostic report with side-by-side XAI & clinical advice
    └── about.html                           # VGG16 + VGG19 hybrid feature fusion & research details
```

---

## ⚙️ Quick Start Guide

### Step 1: Place Your Trained Model File
Copy your saved model from Google Colab into this folder `Chest_xray website/`:
* `pneumonia_Hybrid_VGG16_VGG19_model.keras` **OR**
* `best_hybrid_vgg16_vgg19.keras` **OR**
* `model.keras` / `model.h5`

> *Note: If you run the server before copying the model file, the app automatically initializes the exact VGG16 + VGG19 Hybrid architecture for demonstration purposes.*

---

### Step 2: Install Dependencies
Open PowerShell or Command Prompt in this folder and run:
```bash
pip install -r requirements.txt
```

---

### Step 3: Launch the Web Application
Run:
```bash
python App.py
```
Or simply double-click **`run.bat`**.

Then open your browser and visit:
👉 **`http://127.0.0.1:5000`**

---

## ✨ Features

1. **VGG16 + VGG19 Feature Fusion:**
   - Pre-trained ImageNet backbones concurrently extract low-level spatial and high-level semantic representations.
   - Merged through GlobalAveragePooling2D, Concatenation, Dense(128, ReLU), and Dropout(0.3).

2. **Explainable AI (Grad-CAM):**
   - Visualizes pulmonary opacities, focal consolidations, and infiltrates directly on the radiograph with Jet colormap overlay.

3. **Clinical Decision Support:**
   - Detailed evidence-based advice (CBC, Sputum culture, SpO2 monitoring, follow-up imaging recommendations).

4. **Print & PDF Medical Reports:**
   - One-click print-ready diagnostic summary formatted for clinical case files.

5. **Instant 1-Click Sample Testing:**
   - Built-in test cases for normal lungs and pneumonia-affected lungs.
