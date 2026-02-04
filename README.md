🫁 AI-Powered Lung Disease Detection System
📌 Project Overview
This project is a specialized deep learning diagnostic tool designed to assist medical professionals in detecting lung diseases from Chest X-Ray images.

It utilizes a custom MobileNetV2 architecture enhanced with a CBAM (Convolutional Block Attention Module). This attention mechanism allows the AI to "focus" on specific infected regions of the lung (like lesions or fluid buildup) rather than looking at the entire image generally, significantly improving classification accuracy.

The system is deployed as a secure, high-performance web application featuring a "Dark Medical Mode" interface for optimal viewing in clinical environments.

🏥 Disease Detection Capability
The model is trained to classify X-Rays into four specific categories:

Covid-19 (Viral infection patterns)

Pneumonia (Bacterial/Viral inflammation)

Tuberculosis (TB) (Bacterial infection scarring)

Normal (Healthy lung tissue)

🚀 Key Features
Attention-Based AI: Uses CBAM to prioritize important features in the X-Ray, reducing false positives.

Secure Staff Portal: A restricted-access environment protected by a secure login system (Glassmorphism UI).

Privacy-First Design: All image processing occurs within the session; patient data is processed locally and not stored on external public servers.

Sensitivity Analysis: Provides a real-time probability bar chart, showing the model's confidence levels across all four disease classes.

Robust Image Engine: Automatically handles various image formats (JPEG, PNG) and corrects channel errors (Grayscale vs. RGB) instantly.

🛠️ Tech Stack
Deep Learning Framework: TensorFlow & Keras

Base Architecture: MobileNetV2 (Transfer Learning)

Attention Mechanism: CBAM (Convolutional Block Attention Module)

Web Interface: Streamlit (Python)

Image Processing: NumPy & Pillow (PIL)

📊 Model Performance
Architecture Choice: MobileNetV2 was selected for its lightweight structure, making it ideal for deployment on laptops or edge devices without needing heavy GPU servers.

Enhancement: The addition of the CBAM layer improved feature extraction capabilities compared to the standard base model.

👤 Author
T. Vamsi Krishna Sai

Department: Electronics and Communication Engineering (ECE)

Institute: SRM Institute of Science and Technology (SRMIST)
