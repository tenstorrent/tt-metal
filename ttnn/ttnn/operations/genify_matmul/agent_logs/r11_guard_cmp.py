"""Compare device_kernel_ns of two eval_test_runner test_results.json files (base, new)."""
import json, sys


def load(p):
    return {r["test_name"]: r for r in json.load(open(p))}


b, n = load(sys.argv[1]), load(sys.argv[2])
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
print(f"## guard: name  base_us  new_us  delta  cores(base->new)")
for k in b:
    if k not in n or not b[k].get("device_kernel_ns") or not n[k].get("device_kernel_ns"):
        print(f"{k:80s} missing / {n.get(k, {}).get('status')}")
        continue
    x, y = b[k]["device_kernel_ns"] / 1000, n[k]["device_kernel_ns"] / 1000
    d = (y / x - 1) * 100
    mark = " <<" if d > thr else (" **" if d < -thr else "")
    print(
        f"{k:80s} {x:9.1f} {y:9.1f} {d:+6.1f}%  {int(b[k]['device_num_cores'] or 0)}->{int(n[k]['device_num_cores'] or 0)}{mark}"
    )
