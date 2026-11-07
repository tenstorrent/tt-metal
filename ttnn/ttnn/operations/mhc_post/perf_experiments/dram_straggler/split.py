"""split.py <csv> <run> <out>: keep the header + one run's rows."""
import sys

src, run, out = sys.argv[1:4]
with open(src) as fh, open(out, "w") as fo:
    fo.write(fh.readline())
    hdr = fh.readline()
    fo.write(hdr)
    idx = [h.strip() for h in hdr.split(",")].index("run host ID")
    for l in fh:
        if l.split(",")[idx].strip() == run:
            fo.write(l)
