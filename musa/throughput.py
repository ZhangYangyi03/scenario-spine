"""Throughput of the two paths, reported because a claim about a GPU is worth
measuring even when the answer is that the GPU does not matter.

The corpus is the real one: 30k held-out scenarios, 48 features. Times are wall
clock over 20 repetitions after a warm-up, both paths given the same inputs.
"""
import os, subprocess, sys, time
import numpy as np, torch
import torch_musa  # noqa: F401  -- registers torch.musa
sys.path.insert(0, "/root/work")
from spine import netdef

STATE = "/root/work/musa/state"
z = np.load("/root/autodl-tmp/parity.npz")
Xr = z["Xall"][:30000].astype(np.float32)
mean, std = np.load(f"{STATE}/mean.npy"), np.load(f"{STATE}/std.npy")
X = np.clip((Xr - mean) / std, -8, 8).astype(np.float32)
n, d = X.shape
print("rows", n, "dim", d)

# torch on the device
payload = torch.load("/root/work/musa/state.pt", map_location="cpu")
net = payload["net"].eval()
dev = "musa" if torch.musa.is_available() else "cpu"
net = net.to(dev)
xt = torch.tensor(X).to(dev)
with torch.no_grad():
    for _ in range(3): net(xt[:1000])
    torch.musa.synchronize()
    t0 = time.time()
    for _ in range(20):
        out = net(xt).reshape(-1)
    torch.musa.synchronize()
    tt = (time.time() - t0) / 20
print(f"torch ({dev})   {tt*1000:8.1f} ms for {n} rows -> {n/tt/1000:8.1f}k rows/s")

# the kernel, through the driver
exe = os.path.join("/root/work/musa", "predict")
xp, op = "/tmp/bench_x.f32", "/tmp/bench_y.f32"
with open(xp, "wb") as fh:
    fh.write(X.tobytes())
dn = open(os.devnull, "w")          # kept open across all 21 launches: opening it
                                    # inside the loop would time the open too
subprocess.run([exe, STATE, xp, str(n), str(d), op], stdout=dn, check=True)
t0 = time.time()
for _ in range(20):
    subprocess.run([exe, STATE, xp, str(n), str(d), op], stdout=dn, check=True)
dn.close()
tk = (time.time() - t0) / 20
print(f"kernel (musa)   {tk*1000:8.1f} ms for {n} rows -> {n/tk/1000:8.1f}k rows/s")
r = subprocess.run([exe, STATE, xp, str(n), str(d), op, "100"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "kernel_ms_per_call" in line:
        print("  device-side (event-timed, 100 launches, excludes process start and I/O):", line)
print("the end-to-end kernel time includes process start, malloc, the H2D and D2H copies and "
      "the write to disk; the torch time includes neither a process start nor a "
      "disk write. The comparison is therefore not a kernel-versus-kernel number "
      "and is not quoted as one.")
