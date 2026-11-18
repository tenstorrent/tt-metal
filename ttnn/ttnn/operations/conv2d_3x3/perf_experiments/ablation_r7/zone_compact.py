import sys, re, collections

names = [
    "stem1",
    "stem2",
    "stem3",
    "res2",
    "res3_0",
    "res3_x",
    "res4_0",
    "res4_x",
    "res5_0",
    "res5_1",
    "res5_2",
    "aspp1",
    "aspp2",
    "aspp3",
    "semdec_r3_fc0",
    "instdec_r3_fc0",
    "semdec_r3_fc1",
    "semdec_r2_fc0",
    "instdec_r2_fc0",
    "sem_r2_fc1",
    "inst_r2_fc1",
    "center1",
]
txt = open(sys.argv[1]).read()
blocks = re.split(r"=== run ", txt)[1:]
for i, b in enumerate(blocks):
    lines = b.splitlines()
    hdr = lines[0]
    tk = [l for l in lines if "TRISC-KERNEL" in l]
    tmax = max(float(re.search(r"max\s+([\d.]+)", l).group(1)) for l in tk)
    out = []
    for l in lines:
        m = re.match(r"\s+(\w+)\s+(BRISC|NCRISC|TRISC_0)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)", l)
        if m and float(m.group(3)) >= 0.5:
            out.append(f"{m.group(1)}={m.group(3)}/{m.group(4)}")
    nm = names[i] if i < len(names) else "?"
    print(f"{nm:16s} {hdr.split(',')[0].split(':')[1].strip():10s} T={tmax:6.1f} | " + " ".join(out))
