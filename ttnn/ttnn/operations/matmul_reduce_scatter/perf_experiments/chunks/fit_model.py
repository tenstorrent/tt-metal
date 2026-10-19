import json, glob, itertools, math, sys

R = []
for f in glob.glob(sys.argv[1] + "/*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r.get("median_us") and not r.get("ablate"):
            R.append(r)
CLK = 1.35e3  # cycles per us


def feats(r):
    p = r["plan"]
    M, K, N = r["M"], r["K"], r["N"]
    G = 4
    Kt = K // 32
    C = p["chunks"][0] * p["chunks"][1]
    P = p["P"]
    kb = p["kb"]
    cores = p["grid"][0] * p["grid"][1]
    comp = C * G * P * Kt * 32 / CLK  # padded per-core matmul at HiFi2 peak (us)
    steps = C * G * (Kt // kb)
    tA = 2048
    tW = 1088 if r["sweep"].startswith("kimi") else 2048
    A = M * K // 1024 * tA
    W = K * N // 1024 * tW
    nm, nn = p["chunks"]
    traffic = (A + nm * W) if p["xblock"] == "resident" else (G * nn * A + nm * W)
    dram = traffic / 450e3  # bytes / (B/us) at 450 GB/s
    handoffs = C * (G - 1) * cores / 2  # (core, sending entry) handoffs per port (2 links)
    return dict(comp=comp, steps=steps, dram=dram, handoffs=handoffs, C=C)


def predict(f, eff, a, b, floor=8.0):
    return floor + max(f["comp"] / eff + a * f["steps"], f["dram"], b * f["handoffs"])


best = None
for eff in [x / 100 for x in range(60, 100, 2)]:
    for a in [x / 20 for x in range(0, 41)]:
        for b in [x / 100 for x in range(5, 60, 2)]:
            err = sum(abs(math.log(predict(feats(r), eff, a, b) / r["median_us"])) for r in R) / len(R)
            if best is None or err < best[0]:
                best = (err, eff, a, b)
print("fit: mean |log err| %.3f eff %.2f a(us/step) %.2f b(us/handoff) %.2f" % best, "n =", len(R))
_, eff, a, b = best
for sw in sorted({r["sweep"] for r in R}):
    rs = [r for r in R if r["sweep"] == sw]
    errs = sorted(abs(predict(feats(r), eff, a, b) / r["median_us"] - 1) for r in rs)
    print(f"  {sw:9s} n={len(rs):3d} median err {errs[len(errs)//2]*100:.0f}%  max {errs[-1]*100:.0f}%")


def m_add(f, eff, a, b, c):
    comp = f["comp"] / eff + a * f["steps"]
    return 8 + max(comp + c * f["handoffs"], f["dram"], b * f["handoffs"])


def m_max2(f, eff, a, b, c):
    comp = f["comp"] / eff + a * f["steps"]
    return 8 + max(comp, f["dram"], b * f["handoffs"]) + c * min(comp, b * f["handoffs"])


def fit(model):
    best = None
    for eff in [x / 100 for x in range(60, 100, 3)]:
        for a in [x / 10 for x in range(0, 31)]:
            for b in [x / 100 for x in range(5, 50, 3)]:
                for c in [x / 100 for x in range(0, 60, 5)]:
                    err = sum(abs(math.log(model(feats(r), eff, a, b, c) / r["median_us"])) for r in R) / len(R)
                    if best is None or err < best[0]:
                        best = (err, eff, a, b, c)
    return best


def regret(model, prm):
    shapes = {}
    for r in R:
        shapes.setdefault((r["M"], r["K"], r["N"]), []).append(r)
    out = []
    for k, rs in shapes.items():
        bestm = min(r["median_us"] for r in rs)
        pick = min(rs, key=lambda r: model(feats(r), *prm))
        out.append(
            (
                k,
                pick["plan"]["chunks"],
                pick["plan"]["kb"],
                round(pick["median_us"]),
                round(bestm),
                f"{(pick['median_us']/bestm-1)*100:.1f}%",
            )
        )
    return out


for name, model in (("add", m_add), ("max2", m_max2)):
    b = fit(model)
    print(name, "mean |log err| %.3f eff %.2f a %.2f b %.2f c %.2f" % b)
    print("   ", regret(model, b[1:]))

print("---- model 3: fitted DRAM rate, stream step cost")
FE = [(r, feats(r)) for r in R]


def m3(r, f, eff, a, a_s, b, bw):
    st = r["plan"]["xblock"] == "stream"
    comp = f["comp"] / eff + (a_s if st else a) * f["steps"]
    return 8 + max(comp, f["dram"] * 450 / bw, b * f["handoffs"])


best = None
for eff in [x / 100 for x in range(66, 96, 3)]:
    for a in [x / 10 for x in range(0, 25, 2)]:
        for a_s in [x / 10 for x in range(0, 40, 3)]:
            for b in [x / 100 for x in range(14, 40, 3)]:
                for bw in range(150, 460, 25):
                    err = sum(abs(math.log(m3(r, f, eff, a, a_s, b, bw) / r["median_us"])) for r, f in FE) / len(FE)
                    if best is None or err < best[0]:
                        best = (err, eff, a, a_s, b, bw)
print("m3 mean |log err| %.3f eff %.2f a %.2f a_s %.2f b %.2f bw %d" % best)
prm = best[1:]
shapes = {}
for r, f in FE:
    shapes.setdefault((r["M"], r["K"], r["N"]), []).append((r, f))
for k, rs in shapes.items():
    bestm = min(r["median_us"] for r, _ in rs)
    pick = min(rs, key=lambda rf: m3(rf[0], rf[1], *prm))[0]
    print(
        "  ",
        k,
        "pick",
        pick["plan"]["chunks"],
        "kb",
        pick["plan"]["kb"],
        round(pick["median_us"]),
        "best",
        round(bestm),
        f"regret {(pick['median_us']/bestm-1)*100:.1f}%",
    )
for sw in sorted({r["sweep"] for r in R}):
    errs = sorted(abs(m3(r, f, *prm) / r["median_us"] - 1) for r, f in FE if r["sweep"] == sw)
    print(f"  {sw:9s} median err {errs[len(errs)//2]*100:.0f}%  max {errs[-1]*100:.0f}%")
