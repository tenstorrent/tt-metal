import sys
sys.path.insert(0, "/tmp"); import initwin
# fetchseq.py <cyc> <thread> <elf>: in the last INIT window, each code-line fetch (cycle, line address) and the stall cycles (no new commit)
t = int(sys.argv[2]); r, e, b, d = initwin.window_pcs(sys.argv[3]); w, c, n = initwin.load(sys.argv[1], t, r, e)
fe = [(i, row[5] * 4 if row[5] >= 0 else -1) for i, row in enumerate(w) if row[1]]
commits = set(i for i, p in c)
stall = [i for i in range(len(w)) if i not in commits]
print("window", len(w), "commits", len(c), "stall cycles", len(stall))
print("fetches:", " ".join(f"{i}:{a:#x}" for i, a in fe))
