import csv, sys

path = sys.argv[1]
cx, cy = sys.argv[2].split(",")
with open(path) as fh:
    freq = float(fh.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(fh, skipinitialspace=True))
ns = lambda c: c * 1000.0 / freq
t0 = min(
    int(r["time[cycles since reset]"]) for r in rows if r["zone name"].endswith("-FW") and r["type"] == "ZONE_START"
)
st = {}
ev = []
for r in rows:
    if (r["core_x"], r["core_y"]) != (cx, cy):
        continue
    k = (r["RISC processor type"], r["zone name"])
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        st[k] = t
    elif r["type"] == "ZONE_END" and k in st:
        s = st.pop(k)
        ev.append((ns(s - t0), ns(t - t0), k[0], k[1]))
want = sys.argv[3].split(",") if len(sys.argv) > 3 else None
for s, e, rk, z in sorted(ev):
    if want and rk not in want:
        continue
    if e - s < float(__import__("os").environ.get("MINNS", "300")) and not z.endswith("KERNEL"):
        continue
    print(f"{s:8.0f} {e:8.0f} {e-s:7.0f} {rk:7s} {z}")
