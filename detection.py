# detection.py
# -------------------------------------------------------------
# Deep learning for face recognition in Streamlit (PyTorch)
# - Face detection: MTCNN (facenet-pytorch)
# - Embeddings: InceptionResnetV1 pretrained on VGGFace2
# - Tasks: detection, verification (1:1), identification (1:N)
# - Simple local gallery persisted to disk (embeddings.pkl)
# -------------------------------------------------------------

import os
import pickle
from typing import List, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- Config & Helpers ----------------------
APP_TITLE = "Face Recognition"
GALLERY_PATH = "embeddings.pkl"  # persistent gallery storage


@st.cache_resource(show_spinner=False)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_models(image_size: int = 160):
    device = get_device()
    mtcnn = MTCNN(image_size=image_size, margin=0, device=device, post_process=True)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet


@st.cache_data(show_spinner=False)
def read_image(file) -> Image.Image:
    return Image.open(file).convert("RGB")


# ---------------------- Gallery Utils -------------------------
def load_gallery() -> Dict[str, np.ndarray]:
    if os.path.exists(GALLERY_PATH):
        with open(GALLERY_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_gallery(gallery: Dict[str, np.ndarray]):
    with open(GALLERY_PATH, "wb") as f:
        pickle.dump(gallery, f)


def enroll_embeddings(
    gallery: Dict[str, np.ndarray], label: str, embeddings: List[np.ndarray]
):
    embs = np.vstack(embeddings)
    mean_emb = embs.mean(axis=0, keepdims=True)  # shape (1, 512)
    gallery[label] = mean_emb
    save_gallery(gallery)


# ---------------------- Face Ops ------------------------------
def detect_faces(mtcnn: MTCNN, image: Image.Image):
    boxes, probs = mtcnn.detect(image)
    return boxes, probs


def _ensure_rgb(face: torch.Tensor) -> torch.Tensor:
    """Ensure face tensor has shape (1,3,H,W)."""
    if isinstance(face, torch.Tensor):
        if face.ndim == 2:  # (H,W)
            face = face.unsqueeze(0).repeat(3, 1, 1)  # -> (3,H,W)
        if face.ndim == 3:  # (C,H,W)
            if face.shape[0] == 1:  # grayscale
                face = face.repeat(3, 1, 1)
            face = face.unsqueeze(0)  # -> (1,C,H,W)
        elif face.ndim == 4:  # already batched
            if face.shape[1] == 1:
                face = face.repeat(1, 3, 1, 1)
    return face


def extract_face_tensors(mtcnn: MTCNN, image: Image.Image, boxes: np.ndarray):
    """Extract cropped face tensors from image given bounding boxes."""
    if boxes is None:
        return None

    face_tensors = []
    for box in boxes:
        box = np.array([box], dtype=int).reshape(1, 4)  # (1,4)
        faces = mtcnn.extract(image, box, save_path=None)
        if faces is not None and len(faces) > 0:
            face = _ensure_rgb(faces[0])
            face_tensors.append(face)

    if face_tensors:
        return torch.cat(face_tensors, dim=0)  # (N,3,H,W)
    return None


def extract_single_face_tensor(mtcnn: MTCNN, image: Image.Image, box: List):
    """Extract a single cropped face tensor from image given a bounding box."""
    box = np.array([box], dtype=int).reshape(1, 4)
    faces = mtcnn.extract(image, box, save_path=None)
    if faces is not None and len(faces) > 0:
        return _ensure_rgb(faces[0])  # always (1,3,H,W)
    return None


def get_embeddings(resnet: InceptionResnetV1, faces_tensor: torch.Tensor) -> np.ndarray:
    device = get_device()
    faces_tensor = _ensure_rgb(faces_tensor).to(device)  # <--- safety fix
    with torch.no_grad():
        emb = resnet(faces_tensor).cpu().numpy()  # (N, 512)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb = emb / norms
    return emb


def draw_boxes(
    image: Image.Image,
    boxes: np.ndarray,
    labels: List[str] = None,
    probs: List[float] = None,
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    if boxes is None:
        return img
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        tag = None
        if labels and i < len(labels) and labels[i]:
            tag = labels[i]
        conf = None
        if probs is not None and i < len(probs) and probs[i] is not None:
            conf = f"{probs[i]:.2f}"
        text = ", ".join([t for t in [tag, conf] if t])
        if text:
            tw, th = draw.textlength(text, font=font), 18
            draw.rectangle([x1, y1 - th - 4, x1 + tw + 8, y1], fill=(0, 255, 0))
            draw.text((x1 + 4, y1 - th - 2), text, fill=(0, 0, 0), font=font)
    return img


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0])


# ---------------------- Streamlit UI --------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Detection, verification, and identification using VGGFace2 Model"
)

with st.sidebar:
    st.header("Settings")
    mtcnn_img_size = st.slider("Detector image size", 128, 256, 160, 8)
    verify_threshold = st.slider(
        "Verification threshold (cosine)", 0.50, 0.95, 0.75, 0.01
    )
    top_k = st.slider("Top-K for identification", 1, 5, 3)
    device = get_device()
    st.write(f"**Device:** {'GPU' if device.type == 'cuda' else 'CPU'}")

mtcnn, resnet = load_models(mtcnn_img_size)
gallery = load_gallery()

TAB_DET, TAB_VER, TAB_ID, TAB_ENROLL = st.tabs(
    ["🔍 Detect", "🔗 Verify (1:1)", "👤 Identify (1:N)", "📚 Enroll Gallery"]
)

# ---------------------- Detect Tab ---------------------------
with TAB_DET:
    st.subheader("Face Detection")
    det_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="det"
    )
    if det_file:
        img = read_image(det_file)
        boxes, probs = detect_faces(mtcnn, img)
        if boxes is None:
            st.warning("No faces detected.")
        else:
            out = draw_boxes(img, boxes, probs=probs)
            st.image(out, caption="Detected faces", use_column_width=True)
            faces_tensor = extract_face_tensors(mtcnn, img, boxes)
            if faces_tensor is not None and len(faces_tensor) > 0:
                st.markdown("**Cropped faces**")
                cols = st.columns(min(4, len(faces_tensor)))
                for i, face in enumerate(faces_tensor):
                    fimg = Image.fromarray(
                        (face.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    )
                    cols[i % len(cols)].image(fimg, use_column_width=True)

# ---------------------- Verify Tab ---------------------------
with TAB_VER:
    st.subheader("Face Verification (1:1)")
    c1, c2 = st.columns(2)
    with c1:
        v1 = st.file_uploader("Image A", type=["jpg", "jpeg", "png"], key="ver_a")
    with c2:
        v2 = st.file_uploader("Image B", type=["jpg", "jpeg", "png"], key="ver_b")

    if v1 and v2:
        img1 = read_image(v1)
        img2 = read_image(v2)
        b1, p1 = detect_faces(mtcnn, img1)
        b2, p2 = detect_faces(mtcnn, img2)
        if b1 is None or b2 is None:
            st.error("Could not detect a face in one or both images.")
        else:
            i1 = int(np.argmax(p1))
            i2 = int(np.argmax(p2))
            f1 = extract_single_face_tensor(mtcnn, img1, b1[i1])
            f2 = extract_single_face_tensor(mtcnn, img2, b2[i2])
            
            if f1 is not None and f2 is not None:
                emb1 = get_embeddings(resnet, f1)[0]
                emb2 = get_embeddings(resnet, f2)[0]
                score = cosine_sim(emb1, emb2)
                match = score >= verify_threshold

                st.write(f"**Cosine similarity:** {score:.4f}")
                st.success("MATCH" if match else "NO MATCH")

                c1, c2 = st.columns(2)
                c1.image(
                    draw_boxes(img1, np.array([b1[i1]]), probs=[p1[i1]]),
                    caption="Image A",
                    use_column_width=True,
                )
                c2.image(
                    draw_boxes(img2, np.array([b2[i2]]), probs=[p2[i2]]),
                    caption="Image B",
                    use_column_width=True,
                )
            else:
                st.error("Could not extract faces from one or both images.")

# ---------------------- Identify Tab -------------------------
with TAB_ID:
    st.subheader("Face Identification (1:N)")
    if not gallery:
        st.info("Gallery is empty. Go to 'Enroll Gallery' to add labeled faces.")
    id_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="id"
    )

    if id_file and gallery:
        img = read_image(id_file)
        boxes, probs = detect_faces(mtcnn, img)
        if boxes is None:
            st.warning("No faces detected.")
        else:
            faces_tensor = extract_face_tensors(mtcnn, img, boxes)
            if faces_tensor is not None:
                embs = get_embeddings(resnet, faces_tensor)
                labels = []
                for i in range(embs.shape[0]):
                    e = embs[i]
                    sims = {
                        lab: cosine_sim(e, vec.squeeze(0)) for lab, vec in gallery.items()
                    }
                    top = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]
                    best_label, best_score = top[0]
                    shown = f"{best_label} ({best_score:.2f})"
                    labels.append(shown)
                    with st.expander(f"Face {i+1} – Top {top_k}") as exp:
                        for lab, sc in top:
                            st.write(f"{lab}: {sc:.4f}")
                st.image(
                    draw_boxes(img, boxes, labels=labels, probs=probs), use_column_width=True
                )
            else:
                st.error("Could not extract faces from the image.")

# ---------------------- Enroll Tab ---------------------------
with TAB_ENROLL:
    st.subheader("Enroll / Manage Gallery")

    st.markdown(
        "**Add a person**: enter a label (name) and upload one or more images of the person. We'll average their embeddings."
    )
    new_label = st.text_input("Label (e.g., 'Alice')")
    enroll_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="enroll",
    )
    if new_label and enroll_files:
        embeds = []
        for f in enroll_files:
            img = read_image(f)
            boxes, probs = detect_faces(mtcnn, img)
            if boxes is None:
                st.warning(f"No face found in {f.name}, skipping.")
                continue
            idx = int(np.argmax(probs))
            face_tensor = extract_single_face_tensor(mtcnn, img, boxes[idx])
            if face_tensor is not None:
                emb = get_embeddings(resnet, face_tensor)[0]
                embeds.append(emb)
        if embeds:
            enroll_embeddings(gallery, new_label, embeds)
            st.success(f"Enrolled '{new_label}' with {len(embeds)} image(s).")
        else:
            st.error("Could not extract any embeddings to enroll.")

    st.markdown("---")
    st.markdown("**Current gallery**")
    if gallery:
        for lab in list(gallery.keys()):
            c1, c2 = st.columns([3, 1])
            c1.write(lab)
            if c2.button("Delete", key=f"del_{lab}"):
                gallery.pop(lab, None)
                save_gallery(gallery)
                st.experimental_rerun()
        if st.button("Clear all"):
            gallery.clear()
            save_gallery(gallery)
            st.experimental_rerun()
    else:
        st.info("No enrolled identities yet.")

# ---------------------- Footer -------------------------------
st.markdown(
    """
    <hr/>
    <small>
    Tips: Higher threshold = stricter verification. Typical cosine threshold ~0.7–0.8 (depends on data).
    This demo uses InceptionResnetV1 pretrained on VGGFace2 via <code>facenet-pytorch</code>.
    </small>
    """,
    unsafe_allow_html=True,
)
