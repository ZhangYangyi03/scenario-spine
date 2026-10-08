"""Train the predictor on the corpus and export it for the kernel. Run on the
GPU box; writes into /root/work/musa/state."""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/work")
from spine import dataset as D, learn, netdef

OUT = "/root/autodl-tmp"
STATE = "/root/work/musa/state"
os.makedirs(STATE, exist_ok=True)

tr = np.load(f"{OUT}/train.npz")
X = tr["X"]
yc = tr["ycol"]
Xstd, mean, std = D.standardise(X)
Xstd = np.clip(Xstd, -8, 8)

avail, dev, ver = learn.torch_available()
print("torch", ver, "device", dev, flush=True)
m = learn.train_mlp_torch(Xstd, yc, target="collision", hidden=(256, 256, 128),
                          epochs=40, lr=2e-3, batch=1024, device=dev, seed=0, verbose=True)
print("val_loss", m["val_loss"], "epoch", m["val_epoch"], flush=True)

net = m["net"]
net.eval()
torch.save({"state_dict": net.state_dict(), "net": net}, "/root/work/musa/state.pt")
np.save(f"{OUT}/mean.npy", mean)
np.save(f"{OUT}/std.npy", std)
print("exporting to", STATE, flush=True)
from tools.export_state import export
meta = export("/root/work/musa/state.pt", STATE, mean, std)
print(json.dumps(meta, indent=1), flush=True)

# a small test pack for the parity check: raw features + labels, so the parity
# script can standardise with the exported statistics itself rather than being
# handed already-processed inputs (which would hide a scaler mismatch)
te = np.load(f"{OUT}/test.npz")
np.savez(f"{OUT}/parity.npz", X=te["X"][:2000], ycol=te["ycol"][:2000],
         Xall=te["X"])
print("wrote parity.npz", flush=True)
