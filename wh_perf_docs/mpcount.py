import sys, collections, subprocess, re
sys.path.insert(0, "/tmp"); import initwin
# mpcount.py <cyc> <thread> <elf>: mispredicts in the last INIT window, by the committed pc at that cycle (the branch), with its function
t = int(sys.argv[2]); r, e, b, d = initwin.window_pcs(sys.argv[3]); w, c, n = initwin.load(sys.argv[1], t, r, e)
cnt = collections.Counter(); seg = collections.Counter()
for i, row in enumerate(w):
    if row[2]: cnt[row[0]] += 1; seg[i // 1000] += 1
print("mispredicts per 1000 cycles:", dict(sorted(seg.items())))
for pc, k in cnt.most_common(12):
    s, txt = d.get(pc, ("?", "?")); print(f"  {k:3d}  {pc:#x}  {txt[:45]:45s} {str(s)[:60]}")
