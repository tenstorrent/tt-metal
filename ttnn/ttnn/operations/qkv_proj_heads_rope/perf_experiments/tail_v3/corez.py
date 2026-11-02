"""Per core (last run in the csv): ends (us) of chosen (risc, zone) pairs, sorted by latest kernel end."""
import collections, csv, sys

path = sys.argv[1]
want = [
    ("TRISC_0", "compute_matmul"),
    ("TRISC_2", "compute_last_k_epilogue"),
    ("BRISC", "writer_store"),
    ("NCRISC", "reader_store"),
    ("BRISC", "BRISC-KERNEL"),
    ("NCRISC", "NCRISC-KERNEL"),
] + [tuple(z.split(":")) for z in (sys.argv[3].split(",") if len(sys.argv) > 3 else [])]
with open(path) as f:
    freq = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = [r for r in list(csv.reader(f))[1:] if len(r) >= 12]
run = [r[7].strip() for r in rows if r[10].strip() == "compute_last_k_epilogue"][-1]
rows = [r for r in rows if r[7].strip() == run]
start, end, seq = {}, {}, {}
for r in rows:
    core = (int(r[1]), int(r[2]))
    risc, t, zone, typ = r[3].strip(), int(r[5]), r[10].strip(), r[11].strip()
    k = (core, risc, zone)
    if typ == "ZONE_START":
        start.setdefault(k, t)
        seq.setdefault(k, []).append([t, None])
    elif typ == "ZONE_END":
        end[k] = t
        if k in seq and seq[k][-1][1] is None:
            seq[k][-1][1] = t
t0 = min(v for k, v in start.items() if k[2].endswith("-KERNEL"))
cores = sorted({k[0] for k in end})


def e(c, rz):
    v = end.get((c, *rz))
    return (v - t0) / freq if v else float("nan")


def s(c, rz):
    v = start.get((c, *rz))
    return (v - t0) / freq if v else float("nan")


cores.sort(key=lambda c: -max(e(c, w) for w in want[-2:] if True))
print("core      " + " ".join(f"{z[1][:14]:>14}" for z in want) + "  | starts")
for c in cores[: int(sys.argv[2]) if len(sys.argv) > 2 else 15]:
    print(
        f"{str(c):9} "
        + " ".join(f"{e(c, w):14.2f}" for w in want)
        + "  | "
        + " ".join(f"{s(c, w):6.2f}" for w in want[1:4])
    )
for c in cores[:3]:
    for risc in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
        for z in (
            "st_wait",
            "st_issue",
            "writer_write_barrier",
            "reader_write_barrier",
            "lk_unit",
            "epi_unit",
            "compute_matmul",
            "compute_last_k_epilogue",
        ):
            if (c, risc, z) in seq:
                print(c, risc, z, " ".join(f"[{(a-t0)/freq:.2f},{(b-t0)/freq:.2f}]" for a, b in seq[(c, risc, z)] if b))
