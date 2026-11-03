import itertools
import sys

P = int(sys.argv[1])
LAT = {"mad": 2, "swap": 2, "rnd": 2, "cast": 2, "setexp": 2, "load": 1}
best = []
# explicit issue slots: LMA=0, RN, CA, PHI, P1, P2, SE, LMB  ; macro delays dM (mad), dS (swap), dR (rnd_y), dT (store)
for RN in range(3, 12):
    for CA in range(RN + 1, RN + 5):
        for PHI in range(CA + 1, CA + 5):
            for P1 in range(PHI + 1, PHI + 5):
                for P2 in range(P1 + 1, P1 + 5):
                    for SE in range(P2 + 1, P2 + 5):
                        for LMB in range(SE - 6, SE + 8):
                            slots = [0, RN, CA, PHI, P1, P2, SE, LMB]
                            if len(set(s % P for s in slots)) != 8:
                                continue
                            for dM in range(0, 4):
                                m = 0 + dM + 1
                                for dS in range(dM + 2, 8):
                                    sw = dS + 1
                                    if sw < m + LAT["mad"]:
                                        continue
                                    if RN < sw + LAT["swap"]:
                                        continue
                                    for dR in range(0, 6):
                                        ry = LMB + dR + 1
                                        if ry < SE + LAT["setexp"]:
                                            continue
                                        for dT in range(dR + 2, 8):
                                            st = LMB + dT + 1
                                            if st < ry + 1:
                                                continue
                                            if st - ry >= P:
                                                continue
                                            # units per residue
                                            occ = {}

                                            def add(u, s):
                                                k = (u, s % P)
                                                if k in occ:
                                                    return False
                                                occ[k] = 1
                                                return True

                                            ok = True
                                            for u, s in [
                                                ("mad", m),
                                                ("simple", sw),
                                                ("simple", sw + 1),
                                                ("mad", PHI),
                                                ("mad", P1),
                                                ("mad", P2),
                                                ("simple", SE),
                                                ("round", RN),
                                                ("round", ry),
                                                ("simple", CA),
                                                ("round", CA),
                                                ("store", st),
                                                ("load", 0),
                                                ("load", LMB),
                                            ]:
                                                if not add(u, s):
                                                    ok = False
                                                    break
                                            if not ok:
                                                continue
                                            # register lifetimes (2 sets, period 2P)
                                            # a: written 0 (load), read up to SE ; next write at 2P
                                            if SE >= 2 * P:
                                                continue
                                            # n: written RN, read by ry; next write 2P+RN
                                            if ry >= 2 * P + RN:
                                                continue
                                            # phi: written CA, read P2; LMB dummy-loads into phi at LMB: must be after P2 and before next CA write (2P+CA); and phi not read after LMB
                                            if not (LMB > P2 and LMB < 2 * P + CA):
                                                continue
                                            # explicit-explicit stalls (perf): count
                                            stall = (
                                                max(0, 2 - (CA - RN))
                                                + max(0, 2 - (PHI - CA))
                                                + max(0, 2 - (P1 - PHI))
                                                + max(0, 2 - (P2 - P1))
                                                + max(0, 2 - (SE - P2))
                                            )
                                            span = st
                                            best.append(
                                                (
                                                    stall,
                                                    span,
                                                    dict(
                                                        RN=RN,
                                                        CA=CA,
                                                        PHI=PHI,
                                                        P1=P1,
                                                        P2=P2,
                                                        SE=SE,
                                                        LMB=LMB,
                                                        dM=dM,
                                                        dS=dS,
                                                        dR=dR,
                                                        dT=dT,
                                                        mad=m,
                                                        swap=sw,
                                                        ry=ry,
                                                        st=st,
                                                    ),
                                                )
                                            )
best.sort(key=lambda x: x[0])
print(len(best))
for b in best[:10]:
    print(b)
