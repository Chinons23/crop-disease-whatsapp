"""Plant Disease Classifier - Streamlit app.

Loads the trained model artifact, checks photo quality, and classifies the leaf.

Run:
    streamlit run app.py

Environment variables (optional):
    MODEL_ARTIFACT  Path to the .pt artifact. Defaults to output/models/checkpoints/best_model.pt
"""
import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import v2

# Must be the first Streamlit call.
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")

logger = logging.getLogger("plant_disease_app")

# Configuration
artifact_path = Path(os.environ.get("MODEL_ARTIFACT", "output/models/checkpoints/best_model.pt"))
max_upload_mb = 10
Image.MAX_IMAGE_PIXELS = 50_000_000     # Protects against decompression-bomb images.
top_k = 3
default_confidence_threshold = 0.60


# Image quality gate (same logic and thresholds as the notebook).
class ImageQualityGate:
    """Rule-based photo quality check.

    Args:
        min_brightness, max_brightness (float): Accepted mean grayscale range (0-1).
        min_contrast (float): Minimum grayscale standard deviation.
        min_blur_score (float): Minimum variance of the Laplacian (sharpness).
        min_leaf_fraction (float): Minimum share of green (vegetation) pixels.
    """
    def __init__(
        self,
        min_brightness=0.12,
        max_brightness=0.92,
        min_contrast=0.05,
        min_blur_score=0.002,
        min_leaf_fraction=0.08,
    ):
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.min_blur_score = min_blur_score
        self.min_leaf_fraction = min_leaf_fraction

        # Laplacian kernel for blur detection
        self.laplacian_kernel = torch.tensor(
            [[0., 1., 0.],
             [1., -4., 1.],
             [0., 1., 0.]]
        ).view(1, 1, 3, 3)

    @torch.inference_mode()
    def assess(self, images: torch.Tensor):
        """
        images: float tensor [B, 3, H, W], RGB values in range [0, 1]

        Returns:
            accepted: Boolean tensor [B]
            metrics: Dictionary of per-image quality measures
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected a [B, 3, H, W] tensor, got {tuple(images.shape)}")

        kernel = self.laplacian_kernel.to(images.device)
        red, green, blue = images[:, 0], images[:, 1], images[:, 2]

        # 1. Brightness and contrast
        grayscale = 0.299 * red + 0.587 * green + 0.114 * blue
        brightness = grayscale.mean(dim=(1, 2))
        contrast = grayscale.std(dim=(1, 2))

        # 2. Blur: variance of Laplacian (low variance = few sharp edges).
        laplacian = F.conv2d(grayscale.unsqueeze(1), kernel, padding=1)
        blur_score = laplacian.var(dim=(1, 2, 3))

        # 3. Approximate leaf coverage with Excess Green.
        excess_green = 2 * green - red - blue
        leaf_fraction = (excess_green > 0.08).float().mean(dim=(1, 2))

        brightness_ok = (brightness >= self.min_brightness) & (brightness <= self.max_brightness)
        contrast_ok = contrast >= self.min_contrast
        sharp_enough = blur_score >= self.min_blur_score
        leaf_visible = leaf_fraction >= self.min_leaf_fraction

        accepted = brightness_ok & contrast_ok & sharp_enough & leaf_visible

        metrics = {
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "leaf_fraction": leaf_fraction,
            "brightness_ok": brightness_ok,
            "contrast_ok": contrast_ok,
            "sharp_enough": sharp_enough,
            "leaf_visible": leaf_visible,
        }
        return accepted, metrics

    def get_feedback(self, metrics: dict, index: int = 0) -> list[str]:
        """Turn the failed checks of one image into messages a farmer can act on."""
        messages = []

        if not metrics["brightness_ok"][index].item():
            if metrics["brightness"][index].item() < self.min_brightness:
                messages.append("The photo is too dark. Move to a brighter area.")
            else:
                messages.append("The photo is over-exposed. Avoid strong glare or direct sun.")
        if not metrics["contrast_ok"][index].item():
            messages.append("The photo has very low contrast. Use a plain background.")
        if not metrics["sharp_enough"][index].item():
            messages.append("Hold the phone steady and make sure the leaf is in focus.")
        if not metrics["leaf_visible"][index].item():
            messages.append("Place one affected leaf clearly in the centre of the photo.")

        return messages


# Model loading (runs once per server process, not on every interaction).
@dataclass
class ModelBundle:
    model: nn.Module
    transforms: v2.Compose
    gate: ImageQualityGate
    class_names: list
    mean: list
    std: list
    test_accuracy: float | None
    test_macro_f1: float | None


@st.cache_resource(show_spinner="Loading model...")
def load_bundle(path_str: str) -> ModelBundle:
    """Load the exported artifact and rebuild the model without downloading anything.

    Args:
        path_str (str): Path to the .pt artifact created by the notebook.
    Returns:
        ModelBundle: Model in eval mode plus the preprocessing and gate settings it was trained with.
    Raises:
        FileNotFoundError: if the artifact does not exist.
        ValueError: if the artifact is missing required keys.
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path.resolve()}")

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    # weights_only=True refuses to execute arbitrary pickled code.
    artifact = torch.load(path, map_location="cpu", weights_only=True)

    required = {"model_state_dict", "class_names", "resize_size", "image_size",
                "mean", "std", "quality_gate"}
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Artifact is missing keys: {sorted(missing)}")

    class_names = artifact["class_names"]

    # Rebuild the same architecture used in training (weights=None -> no internet needed).
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=len(class_names), bias=True))
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    # Identical to the notebook's validation/test preprocessing.
    transforms = v2.Compose([
        v2.Resize(size=artifact["resize_size"], antialias=True),
        v2.CenterCrop(size=(artifact["image_size"], artifact["image_size"])),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=artifact["mean"], std=artifact["std"]),
    ])

    return ModelBundle(
        model=model,
        transforms=transforms,
        gate=ImageQualityGate(**artifact["quality_gate"]),
        class_names=class_names,
        mean=artifact["mean"],
        std=artifact["std"],
        test_accuracy=artifact.get("test_accuracy"),
        test_macro_f1=artifact.get("test_macro_f1"),
    )


# Helpers
def read_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an upright RGB PIL image.

    Args:
        data (bytes): Raw file content.
    Returns:
        Image.Image: RGB image with the camera's EXIF rotation applied.
    Raises:
        ValueError: if the file is too large, not an image, or corrupt.
    """
    if len(data) > max_upload_mb * 1024 * 1024:
        raise ValueError(f"The file is larger than {max_upload_mb} MB. Please use a smaller photo.")
    try:
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image)      # Phone photos are often stored rotated.
        return image.convert("RGB")                 # Handles PNG alpha, CMYK, grayscale.
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError) as error:
        raise ValueError("This file could not be read as an image. Please upload a JPG or PNG photo.") from error


def denormalize(images: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """Undo Normalize so the quality gate receives values in [0, 1]."""
    mean_t = torch.tensor(mean, device=images.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=images.device).view(-1, 1, 1)
    return (images * std_t + mean_t).clamp(0, 1)


def format_label(class_name: str):
    """Split 'Tomato___Late_blight' into ('Tomato', 'Late blight', is_healthy)."""
    crop, _, condition = class_name.partition("___")
    clean = lambda text: " ".join(text.replace("_", " ").split())
    crop, condition = clean(crop), clean(condition) or "Unknown"
    is_healthy = condition.lower() == "healthy"
    return crop, condition[:1].upper() + condition[1:], is_healthy


def analyse(image: Image.Image, bundle: ModelBundle):
    """Preprocess one image and run the quality gate and the classifier.

    Returns:
        tuple: (accepted, feedback messages, quality metrics, top-k list of (class_name, probability))
    """
    batch = bundle.transforms(image).unsqueeze(0)      # [1, 3, H, W], normalised

    accepted, metrics = bundle.gate.assess(denormalize(batch, bundle.mean, bundle.std))
    feedback = bundle.gate.get_feedback(metrics, 0)

    with torch.inference_mode():
        probabilities = torch.softmax(bundle.model(batch), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(top_k, len(bundle.class_names)))
    top = [(bundle.class_names[i], v) for i, v in zip(indices.tolist(), values.tolist())]

    return bool(accepted[0].item()), feedback, metrics, top


# User interface
def main() -> None:
    st.title("Plant Disease Classifier")
    st.caption("Upload or capture a photo of a single leaf to get a suggested diagnosis.")

    try:
        bundle = load_bundle(str(artifact_path))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.exception("Could not load model")
        st.error(f"The model could not be loaded. {error}")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        confidence_threshold = st.slider(
            "Minimum confidence", 0.30, 0.95, default_confidence_threshold, 0.05,
            help="Predictions below this confidence are flagged as uncertain.")
        show_details = st.checkbox("Show image-quality details", value=False)

        st.header("About the model")
        if bundle.test_accuracy is not None:
            st.write(f"Test accuracy: **{bundle.test_accuracy:.1f}%**")
        if bundle.test_macro_f1 is not None:
            st.write(f"Test macro-F1: **{bundle.test_macro_f1:.3f}**")
        crops = sorted({format_label(name)[0] for name in bundle.class_names})
        with st.expander(f"Supported crops ({len(crops)})"):
            st.write(", ".join(crops))
        st.caption("Scores come from PlantVillage test images (single leaf, plain background). "
                   "Real field photos are usually harder.")

    # Input
    upload_tab, camera_tab = st.tabs(["Upload photo", "Use camera"])
    with upload_tab:
        uploaded_file = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"])
    with camera_tab:
        camera_file = st.camera_input("Take a photo of one leaf")

    source = camera_file or uploaded_file
    if source is None:
        st.info("Add a photo to begin. Tips: one leaf, centred, in focus, in good daylight.")
        return

    try:
        image = read_image(source.getvalue())
    except ValueError as error:
        st.error(str(error))
        return

    image_col, result_col = st.columns([1, 1], gap="large")
    with image_col:
        st.image(image, caption="Your photo")

    with result_col:
        try:
            with st.spinner("Analysing..."):
                accepted, feedback, metrics, top = analyse(image, bundle)
        except Exception:                      # Keep the app alive and the error out of the user's face.
            logger.exception("Inference failed")
            st.error("Something went wrong while analysing this photo. Please try another one.")
            return

        # Quality gate
        if not accepted:
            st.warning("**Please retake the photo.**\n\n" + "\n".join(f"- {m}" for m in feedback))
            if not st.checkbox("Analyse this photo anyway (results may be unreliable)"):
                if show_details:
                    st.json({k: round(v[0].item(), 4) for k, v in metrics.items() if v.dtype.is_floating_point})
                return

        # Result
        best_name, best_prob = top[0]
        crop, condition, is_healthy = format_label(best_name)

        st.subheader(f"{crop}: {condition}")
        if best_prob < confidence_threshold:
            st.warning(f"The model is not confident ({best_prob:.0%}). Retake the photo or ask an "
                       "agricultural extension officer to confirm.")
        elif is_healthy:
            st.success(f"The leaf looks healthy ({best_prob:.0%} confidence).")
        else:
            st.error(f"Likely disease detected ({best_prob:.0%} confidence).")

        st.markdown("**Top predictions**")
        for class_name, probability in top:
            crop_name, condition_name, _ = format_label(class_name)
            st.progress(float(probability), text=f"{crop_name}: {condition_name} ({probability:.1%})")

        if show_details:
            with st.expander("Image-quality details", expanded=True):
                st.json({k: round(v[0].item(), 4) for k, v in metrics.items() if v.dtype.is_floating_point})

    st.divider()
    st.caption("This tool supports, but does not replace, advice from a trained agronomist. "
               "It only knows the crops and diseases listed in the sidebar; a leaf from any "
               "other crop will still be forced into one of those classes.")


main()
