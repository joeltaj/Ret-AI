"""Quick backend smoke test (no GUI)."""
import json
import os

import inference
import xai

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ckpt = os.path.join(ROOT, "best_strict_cbm.pt")
kb_path = os.path.join(ROOT, "fgadr_concept_kb.yaml")
img = os.path.join(ROOT, "sample_fundus.jpg")

print("Device:", inference.DEVICE)
print("Loading model...")
model = inference.load_model(ckpt)
print("Model loaded. Running inference...")
result = inference.predict(model, img)
print(json.dumps(result, indent=2))

kb = xai.load_kb(kb_path)
expl = xai.narrate(result, kb)
expl.pop("retrieved_knowledge", None)
print("\n--- EXPLANATION ---")
print(json.dumps(expl, indent=2))
