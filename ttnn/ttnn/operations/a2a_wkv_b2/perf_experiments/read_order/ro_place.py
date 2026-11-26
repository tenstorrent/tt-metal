"""read_order: build an RO_CORES string from per-row class counts (logical rows 0..7 = phys y 2..9), each row's
classes spread evenly over its 12 columns. Usage: ro_place.py "L12|L12|P3M3A3B3|..." (8 rows, '|' separated)"""
import sys, re


def row_string(spec, width=12):
    counts = [(c, int(k)) for c, k in re.findall(r"([LPMABX])(\d+)", spec)]
    assert sum(k for _, k in counts) == width, spec
    # even spread: each class item gets ideal positions (i + 0.5) * width / k; sort by that
    items = []
    for ci, (c, k) in enumerate(counts):
        items += [((i + 0.5) * width / k, ci, c) for i in range(k)]
    return "".join(c for _, _, c in sorted(items))


def build(spec):
    rows = spec.split("|")
    assert len(rows) == 8
    return "".join(row_string(r) for r in rows)


if __name__ == "__main__":
    print(build(sys.argv[1]))
