exec(open("/tmp/dfit.py").read().split("res=[]")[0])


def pipe_at(b, B, d):
    return b + 1 < B and (d >= 3 or b + d >= B)


def score(p, h0, h1, rho, s1, only_d2=True):
    B, d, k, G = p["B"], p["d"], p["kmax"], p["w"]
    H = h0 + h1 * G
    c = B * k + H + k / 4 - (s1 if B == 1 else 0)
    for b in range(B - 1):
        if d == 1:
            c += 1e6
        elif pipe_at(b, B, d):
            c += max(0.0, H - rho * k)
        else:
            c += max(0.0, H - (1 - rho) * k)
    return c


rhos = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
for s1 in (0, 4, 8, 12, 16):
    print("s1", s1, "h0 \\ rho", rhos)
    for h0 in range(16, 48, 2):
        print(f"{h0:3d}", " ".join(f"{evaluate(h0,0.75,r,s1)[0]:5.2f}/{evaluate(h0,0.75,r,s1)[1]:4.2f}" for r in rhos))
print("=== verbose 30 .75 .35 12")
evaluate(30, 0.75, 0.35, 12, verbose=True)
