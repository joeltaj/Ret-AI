"""
Inference engine for the DR-grading desktop app.

Loads a trained strict-CBM checkpoint and runs a single-image prediction,
returning the DR grade plus per-concept probabilities.

Preprocessing matches the *eval* transform used during training in fusion.py
(CLAHE on the LAB L-channel, then Resize -> ToTensor -> Normalize). Using the
same transform as training is important for prediction accuracy.
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile

from model_def import DRConceptFusionModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional: CLAHE requires OpenCV. If unavailable we fall back gracefully.
try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The strict-CBM checkpoints in this project use these 6 concepts.
CONCEPT_COLS = ["MA", "HE", "EX", "SE", "IRMA", "NV"]

CONCEPT_NAMES = {
    "MA": "Microaneurysm",
    "HE": "Hemorrhage",
    "EX": "Hard exudate",
    "SE": "Soft exudate / cotton-wool spot",
    "IRMA": "Intraretinal microvascular abnormality",
    "NV": "Neovascularization",
}

GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 512


class _CLAHE_LAB:
    """Apply CLAHE on the L-channel in LAB colour space (matches training)."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img_clahe)
        return img


def build_transform(use_clahe=True):
    steps = []
    if use_clahe and _HAS_CV2:
        steps.append(_CLAHE_LAB())
    steps += [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ]
    return T.Compose(steps)


def load_model(checkpoint_path, backbone_name="densenet121"):
    """Load a strict-CBM checkpoint into eval mode.

    n_quant_features is set to 1 only to satisfy the constructor's validation;
    strict_cbm mode never builds the quantitative encoder, so the value is
    otherwise unused and no quant params are expected in the checkpoint.
    """
    model = DRConceptFusionModel(
        n_concepts=len(CONCEPT_COLS),
        n_quant_features=1,
        n_classes=len(GRADE_NAMES),
        mode="strict_cbm",
        backbone_name=backbone_name,
        pretrained=False,
        freeze_backbone=False,
        dropout_rate=0.3,
        residual_scale=0.5,
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict(model, image, threshold=0.5):
    """Run inference on a PIL image or path. Returns a result dict."""
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    transform = build_transform()
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor, None)

    concept_prob = output["concept_probs"].cpu().numpy()[0]
    grade_prob = torch.softmax(output["task_logits"], dim=1).cpu().numpy()[0]
    grade = int(np.argmax(grade_prob))

    concepts = {c: float(p) for c, p in zip(CONCEPT_COLS, concept_prob)}
    active = [c for c, p in concepts.items() if p >= threshold]

    return {
        "grade": grade,
        "grade_name": GRADE_NAMES[grade],
        "grade_probability": float(grade_prob[grade]),
        "grade_distribution": {
            GRADE_NAMES[i]: float(p) for i, p in enumerate(grade_prob)
        },
        "concepts": concepts,
        "active_concepts": active,
        "threshold": threshold,
    }
