"""DINOv2-base CLS features for every saved policy frame (non-privileged vision baseline).
Writes data/feats/<run>/ep#####.npy (T x 768, float16). Skips episodes already done."""
import glob, os, sys
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

RUNS = sys.argv[1:] or ["c0", "switch", "switch_late", "negation", "negation2", "novel"]
B = os.path.join(os.path.dirname(__file__), "..", "data")
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base", torch_dtype=torch.float16).cuda().eval()
mean = torch.tensor(proc.image_mean, device="cuda").view(1, 3, 1, 1)
std = torch.tensor(proc.image_std, device="cuda").view(1, 3, 1, 1)
for run in RUNS:
    os.makedirs(os.path.join(B, "feats", run), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(B, "runs", run, "img*.npz"))):
        out = os.path.join(B, "feats", run, os.path.basename(f).replace("img", "ep").replace(".npz", ".npy"))
        if os.path.exists(out):
            continue
        imgs = np.load(f)["img"]
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs), 128):
                x = torch.tensor(imgs[i:i + 128], device="cuda").permute(0, 3, 1, 2).float() / 255.0
                x = ((x - mean) / std).half()
                feats.append(model(pixel_values=x).pooler_output.float().cpu().numpy())
        np.save(out, np.concatenate(feats).astype(np.float16))
    print(run, len(glob.glob(os.path.join(B, "feats", run, "*.npy"))), flush=True)
