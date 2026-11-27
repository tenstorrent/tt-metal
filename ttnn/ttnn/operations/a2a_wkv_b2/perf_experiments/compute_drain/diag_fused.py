"""Diagnose the fused (tilize-in-unpack) output: for each output row of tile row 0, find which 16-datum RM
segments (same offset for every K tile k, + k*32 datums) feed SrcB face columns 0-15 / 16-31."""


def main():
    import torch

    torch.manual_seed(0)
    attn = torch.randn(1, 1, 160, 512).to(torch.bfloat16)
    w = (torch.randn(1, 1, 512, 128) * 0.05).to(torch.bfloat16)
    out = torch.load("results/out_fused.pt").float()[0, 0]
    A = attn[0, 0].float()
    W = w[0, 0].float()
    flat = A.flatten()
    cands = list(range(0, 32 * 512, 16))

    def contrib(off, h):
        acc = torch.zeros(128)
        for k in range(16):
            s = k * 32 + off
            acc += flat[s : s + 16] @ W[32 * k + 16 * h : 32 * k + 16 * h + 16]
        return acc

    M0 = torch.stack([contrib(o, 0) for o in cands])
    M1 = torch.stack([contrib(o, 1) for o in cands])
    for i in [0, 1, 2, 3, 8, 15, 16, 17, 31]:
        t = out[i]
        s0 = (M0 @ t) / (M0.norm(dim=1))
        s1 = (M1 @ t) / (M1.norm(dim=1))
        t0 = torch.topk(s0, 30).indices
        t1 = torch.topk(s1, 30).indices
        best = None
        for a in t0:
            for b in t1:
                e = (M0[a] + M1[b] - t).norm().item()
                if best is None or e < best[0]:
                    best = (e, cands[a], cands[b])
        print(
            i,
            "err %.3f/%.2f" % (best[0], t.norm()),
            "c0-15 <- row %d col %d" % divmod(best[1], 512),
            " c16-31 <- row %d col %d" % divmod(best[2], 512),
        )


if __name__ == "__main__":
    main()
