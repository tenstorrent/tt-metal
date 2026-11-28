import importlib, time
import numpy as np

torch = importlib.import_module("torch")  # experiment check, not ttnn package code
sig = np.float32(65537.0)
bad = 0
t = time.time()
step = 1 << 26
for s in range(-(1 << 31), 1 << 31, step):
    n = np.arange(s, s + step, dtype=np.int64).astype(np.int32)
    x = n.astype(np.float32)  # RNE
    g = (x * sig).astype(np.float32)
    d = (x - g).astype(np.float32)
    h = (g + d).astype(np.float32)
    want = torch.from_numpy(x).to(torch.bfloat16).to(torch.float32).numpy()
    b = h.view(np.uint32) != want.view(np.uint32)
    if b.any():
        bad += int(b.sum())
        if bad < 50:
            print("bad e.g.", n[b][:5], h[b][:5], want[b][:5])
print("total bad", bad, time.time() - t)
