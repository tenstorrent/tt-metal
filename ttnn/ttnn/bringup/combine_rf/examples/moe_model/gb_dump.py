import json

from global_block import *

R = {}
for name, p in M.items():
    for st in ("4x4", "8x4"):
        SP, TP, wtp, wsp = STAGES[st]
        for T in (5120, 16384):
            for tie in (False, True):
                b = sim(SP, TP, wtp, wsp, p["E"], p["K"], min(T, 8192), tie_split=tie)
                b = {k: v * T / min(T, 8192) for k, v in b.items()}
                for lev in (("tie",), ("tie", "fp8")) if tie else ((),):
                    r = block(p, st, T, b, lev)
                    R["|".join([name, st, str(T), "+".join(lev) or "today"])] = r
json.dump(R, open("gb.json", "w"))
