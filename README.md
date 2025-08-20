# Face-Detection-with-VGGface2
Deep learning for face recognition: detect face in photographs, face identification &amp; verification with VGGface2

# 🧑‍💻 Face Recognition with VGGFace2 (Streamlit App)

This project is a **Streamlit-based Face Recognition system** built using **PyTorch** and **facenet-pytorch**.  
It provides an easy-to-use interface for **face detection, verification (1:1), and identification (1:N)** with a persistent **local face gallery**.

---

## 🚀 Features
- **Face Detection**: Detect faces in an image using MTCNN.
- **Verification (1:1)**: Check if two faces belong to the same person.
- **Identification (1:N)**: Match a face against a gallery of known identities.
- **Gallery Management**: Enroll new identities, delete, or clear the gallery (stored in `embeddings.pkl`).
- **Embeddings**: Extracted using `InceptionResnetV1` pretrained on **VGGFace2**.
- **Web App**: Built with **Streamlit** for interactive UI.

---
