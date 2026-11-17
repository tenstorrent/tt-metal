import csv, json, sys

rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[0]
ci = hdr.index("DEVICE KERNEL DURATION [ns]")
oc = hdr.index("OP CODE")
ops = [r for r in rows[1:] if "Generic" in r[oc]]
recs = {}
for l in open(sys.argv[2]):
    r = json.loads(l)
    recs[r["idx"]] = r  # last wins (pass 1 of up-front collect runs bodies neutered)
recs = [recs[k] for k in sorted(recs)]
print(f"generic ops: {len(ops)}  cases: {len(recs)}")
for r, o in zip(recs, ops):
    keys = ("out_dtype", "fp32_dest", "natural_pairs", "block_size", "unpack_to_dest")
    print(
        f"{r['idx']:2d} {r['variant']:10s} ns={o[ci]:>10s}  "
        + " ".join(f"{k}={r[k]}" for k in keys)
        + (
            f"  pcc={r['pcc']:.6f} rms={r['rel_rms']:.5f} maxulp={r['max_ulp']:.2f} meanulp={r['mean_ulp']:.3f} >0.5ulp={r['frac_gt_0p5ulp']:.4f} >1ulp={r['frac_gt_1ulp']:.5f} tiny_abs={r.get('tiny_max_abs',0):.2e} ulp32max={r.get('max_ulp32',0):.0f} ulp32mean={r.get('mean_ulp32',0):.0f}"
            if "pcc" in r
            else ""
        )
    )
