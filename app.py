import io

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import SegformerForSemanticSegmentation


# -----------------------------
# Configuration
# -----------------------------

MODEL_ID = "ChillinSpyder/ViT-image-segmentation-model"

LONG_SIDE = 512

CLASS_NAMES = [
    "other",
    "wall",
    "road",
    "vegetation",
    "vehicle",
    "roof",
    "water",
]

PALETTE = np.array(
    [
        (80, 80, 80),
        (230, 160, 60),
        (190, 190, 200),
        (50, 160, 60),
        (220, 50, 50),
        (150, 60, 40),
        (50, 110, 220),
    ],
    dtype=np.uint8,
)

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# -----------------------------
# Load model once
# -----------------------------

@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_ID
    )

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()


# -----------------------------
# Preprocessing
# -----------------------------

def prepare_image(content: bytes) -> np.ndarray:

    try:
        image = Image.open(io.BytesIO(content))

        width, height = image.size

        scale = min(
            1.0,
            LONG_SIDE / max(width, height)
        )

        size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )

        image = (
            image
            .convert("RGB")
            .resize(size, Image.Resampling.LANCZOS)
        )

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as err:

        raise ValueError(
            "The file is not a readable image."
        ) from err

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = array / 255.0

    array = (array - MEAN) / STD

    array = array.transpose(2, 0, 1)

    return np.ascontiguousarray(array)


# -----------------------------
# Segmentation
# -----------------------------

def segment(content: bytes) -> np.ndarray:

    tensor = torch.from_numpy(
        prepare_image(content)
    ).unsqueeze(0).to(device)

    with torch.inference_mode():

        logits = model(pixel_values=tensor).logits                     # (1, C, H/4, W/4)
        logits = F.interpolate(logits, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        mask = logits.argmax(dim=1)[0]

    return (
        mask
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


# -----------------------------
# Colorize
# -----------------------------

def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Drone Image Segmentation",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Drone Image Semantic Segmentation")

st.write(
    "Upload a drone image and segment it into "
    "road, vegetation, vehicles, buildings, water, etc."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    original = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    st.subheader("Input Image")

    st.image(
        original,
        width = 300,
    )

    if st.button(
        "🚀 Segment Image",
        type="primary",
    ):

        with st.spinner("Running segmentation..."):

            mask = segment(image_bytes)

            result = colorize(mask)

        st.subheader("Segmentation")

        st.image(
            result,
            use_container_width=True,
        )

        # Download result
        output = io.BytesIO()

        Image.fromarray(result).save(
            output,
            format="PNG",
        )

        st.download_button(
            "⬇️ Download Segmentation",
            data=output.getvalue(),
            file_name="segmentation.png",
            mime="image/png",
        )

        st.success("Segmentation complete!")

        st.write(
            "Device:",
            str(device)
        )
