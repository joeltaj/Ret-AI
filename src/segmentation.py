"""
Lesion segmentation (U-Net) + colour overlay for the DR-grading app.

Model: segmentation_models_pytorch U-Net, resnet34 encoder, 6 lesion channels
(MA, HE, EX, SE, IRMA, NV) — multi-label (independent sigmoid per channel).
Checkpoint: ../best_unet_multilesion.pt (stores lesions/encoder/image_size).

Preprocessing matches training in segmen.py:
    CLAHE(LAB L-channel) -> resize(size) -> /255 -> normalize(ImageNet mean/std)
"""

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# RGB colours used to paint each lesion in the overlay.
LESION_COLORS = {
    "MA": (255, 82, 82),     # red
    "HE": (255, 145, 0),     # deep orange
    "EX": (255, 235, 59),    # yellow
    "SE": (0, 230, 118),     # green
    "IRMA": (41, 182, 246),  # light blue
    "NV": (213, 0, 249),     # magenta
}


def _clahe(pil_img):
    if not _HAS_CV2:
        return pil_img
    img_np = np.array(pil_img)
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    return pil_img


def load_segmenter(checkpoint_path):
    """Load the U-Net checkpoint. Returns (model, lesions, image_size)."""
    ck = torch.load(checkpoint_path, map_location=DEVICE)
    lesions = ck.get("lesions", ["MA", "HE", "EX", "SE", "IRMA", "NV"])
    encoder = ck.get("encoder", "resnet34")
    image_size = int(ck.get("image_size", 512))

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=len(lesions),
    )
    state_dict = ck.get("model_state_dict", ck.get("state_dict", ck))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, lesions, image_size


def predict_masks(model, image, lesions, image_size=512):
    """Run segmentation on a PIL image or path.

    Returns a dict: lesion_id -> probability map (float32, HxW) at the ORIGINAL
    image resolution, so it can be overlaid directly on the source image.
    """
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    orig_w, orig_h = image.size
    proc = _clahe(image)
    arr = np.asarray(proc)

    if _HAS_CV2:
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    else:
        arr = np.asarray(proc.resize((image_size, image_size), Image.BILINEAR))

    arr = arr.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    tensor = torch.from_numpy(
        np.ascontiguousarray(arr.transpose(2, 0, 1))
    ).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # (C, size, size)

    out = {}
    for i, lesion in enumerate(lesions):
        pmap = probs[i]
        if _HAS_CV2:
            pmap = cv2.resize(pmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            pmap = np.asarray(
                Image.fromarray((pmap * 255).astype(np.uint8)).resize(
                    (orig_w, orig_h), Image.BILINEAR
                ),
                np.float32,
            ) / 255.0
        out[lesion] = pmap.astype(np.float32)
    return out


def build_overlay(image, prob_maps, active_lesions=None, threshold=0.5, alpha=0.45):
    """Blend coloured lesion masks over the original image.

    `active_lesions`: iterable of lesion ids to draw (None = all present).
    Returns an RGB PIL image the same size as `image`.
    """
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    base = np.asarray(image, np.float32).copy()
    if active_lesions is None:
        active_lesions = list(prob_maps.keys())

    for lesion in active_lesions:
        pmap = prob_maps.get(lesion)
        if pmap is None:
            continue
        mask = pmap >= threshold
        if not mask.any():
            continue
        color = np.array(LESION_COLORS.get(lesion, (255, 255, 255)), np.float32)
        base[mask] = (1 - alpha) * base[mask] + alpha * color

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def lesion_pixel_summary(prob_maps, threshold=0.5):
    """Return {lesion: fraction of image pixels above threshold}."""
    summary = {}
    for lesion, pmap in prob_maps.items():
        summary[lesion] = float((pmap >= threshold).mean())
    return summary


# ---------------------------------------------------------------------------
# Connected-component analysis, box rendering, segmentation map, quant metrics
# ---------------------------------------------------------------------------
def _as_pil(image):
    if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


def _min_area(shape):
    """Ignore specks: min lesion component area scaled to image size."""
    h, w = shape[:2]
    return max(12, int(3e-6 * h * w))


def lesion_components(prob_maps, active_lesions=None, threshold=0.5):
    """Connected components per lesion.

    Returns list of dicts: {lesion, x, y, w, h, area, cx, cy}. Requires OpenCV;
    returns [] if OpenCV is unavailable.
    """
    if not _HAS_CV2 or not prob_maps:
        return []
    if active_lesions is None:
        active_lesions = list(prob_maps.keys())

    comps = []
    for lesion in active_lesions:
        pmap = prob_maps.get(lesion)
        if pmap is None:
            continue
        mask = (pmap >= threshold).astype(np.uint8)
        if not mask.any():
            continue
        min_area = _min_area(mask.shape)
        n, _labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):  # skip background label 0
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            comps.append({
                "lesion": lesion,
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "area": area,
                "cx": float(cents[i, 0]),
                "cy": float(cents[i, 1]),
            })
    return comps


def render_detection(image, prob_maps, active_lesions=None, threshold=0.5):
    """Original image with coloured bounding boxes around lesion components."""
    image = _as_pil(image)
    arr = np.asarray(image).copy()
    if not _HAS_CV2 or not prob_maps:
        return Image.fromarray(arr)
    thick = max(2, arr.shape[1] // 350)
    for comp in lesion_components(prob_maps, active_lesions, threshold):
        r, g, b = LESION_COLORS.get(comp["lesion"], (255, 255, 255))
        cv2.rectangle(
            arr, (comp["x"], comp["y"]),
            (comp["x"] + comp["w"], comp["y"] + comp["h"]),
            (int(r), int(g), int(b)), thick,
        )
    return Image.fromarray(arr)


def render_segmentation_map(prob_maps, size, active_lesions=None, threshold=0.5):
    """Coloured lesion masks on a black background (segmentation map view)."""
    if not prob_maps:
        return Image.new("RGB", (size, size), (0, 0, 0))
    if active_lesions is None:
        active_lesions = list(prob_maps.keys())
    ref = next(iter(prob_maps.values()))
    canvas = np.zeros((ref.shape[0], ref.shape[1], 3), np.uint8)
    for lesion in active_lesions:
        pmap = prob_maps.get(lesion)
        if pmap is None:
            continue
        mask = pmap >= threshold
        if mask.any():
            canvas[mask] = LESION_COLORS.get(lesion, (255, 255, 255))
    return Image.fromarray(canvas)


def _retina_geometry(image):
    """Estimate retina mask area and centre from the fundus image."""
    arr = np.asarray(_as_pil(image))
    gray = arr.mean(axis=2)
    retina = gray > 12
    area = int(retina.sum())
    if area == 0:
        h, w = gray.shape
        return h * w, w / 2.0, h / 2.0, min(h, w) / 2.0
    ys, xs = np.nonzero(retina)
    cx, cy = float(xs.mean()), float(ys.mean())
    radius = float(np.sqrt(area / np.pi))
    return area, cx, cy, radius


def _location_label(cx, cy, retina_cx, retina_cy, radius):
    dx = (cx - retina_cx) / max(radius, 1.0)
    dy = (cy - retina_cy) / max(radius, 1.0)
    if (dx * dx + dy * dy) ** 0.5 < 0.33:
        return "Central (macula region)"
    vertical = "Superior" if dy < 0 else "Inferior"
    horizontal = "left" if dx < 0 else "right"
    return f"{vertical}-{horizontal} (image-relative)"


def quantitative_summary(image, prob_maps, active_lesions=None, threshold=0.5):
    """Real lesion measurements from the segmentation masks (pixels / %).

    No physical (mm) calibration is assumed; areas are reported in pixels and
    as a percentage of the estimated retina area.
    """
    comps = lesion_components(prob_maps, active_lesions, threshold)
    retina_area, rcx, rcy, radius = _retina_geometry(image)

    n = len(comps)
    total_area = int(sum(c["area"] for c in comps))
    avg_area = float(total_area / n) if n else 0.0
    area_pct = 100.0 * total_area / retina_area if retina_area else 0.0

    per_lesion = {}
    for c in comps:
        d = per_lesion.setdefault(c["lesion"], {"count": 0, "area_px": 0})
        d["count"] += 1
        d["area_px"] += c["area"]

    if comps:
        wsum = float(sum(c["area"] for c in comps))
        gx = sum(c["cx"] * c["area"] for c in comps) / wsum
        gy = sum(c["cy"] * c["area"] for c in comps) / wsum
        location = _location_label(gx, gy, rcx, rcy, radius)
    else:
        location = "—"

    return {
        "n_lesions": n,
        "total_area_px": total_area,
        "total_area_pct": area_pct,
        "avg_area_px": avg_area,
        "location": location,
        "per_lesion": per_lesion,
        "retina_area_px": retina_area,
    }
