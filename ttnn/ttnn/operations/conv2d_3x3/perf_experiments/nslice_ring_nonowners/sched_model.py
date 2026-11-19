# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Schedule model of one grid row of the BLOCK N-slice ring walk (host-only, no ttnn).

Two levels:
  1. step model (each real step = 1 unit, a step's pack lands L after the producing step's): the lower bound argument
     and the non-bit-exact variants the brief floated (pre-reduce / hand-off / spread);
  2. a row-level discrete-event model of HEAD's real pipeline (math one DEST half ahead, packer gated by the ring token,
     reader writes + arrival atomics, 2 writes in flight) with the forwarding granularity as a knob: group tokens
     (HEAD) vs tile-row tokens (candidate). Calibrated against round-10 aspp.convs.3 row y=5 (packer walk end
     col 8 / 9 / 10 / 0 = 20.8 / 21.7 / 22.5 / 23.4 us, a ~13 us walk on col 8).

    python3 sched_model.py
"""

import itertools


# ---------------------------------------------------------------- step model
def step_model(k, owners, L, order="head", prereduce=None):
    """Makespan per column. order 'head': column c walks owners c-1, c-2, ... (skipping non-owners), each chain visits
    every column in ring order. A step's end >= its column's previous end + 1 and >= the chain's previous visit end + L.
    """
    fin = {}
    end = {}
    for _ in range(400):
        for c in range(k):
            t = 0.0
            for j in range(k):
                o = (c - 1 - j) % k
                if o not in owners:
                    continue
                e = t + 1
                if j >= 1:
                    # previous visit of chain o: column c-1 at its step j-1 (same chain)
                    prev = fin.get(((c - 1) % k, j - 1), float("inf"))
                    e = max(e, prev + L)
                fin[(c, j)] = e
                t = e
            end[c] = t
    return end


def owner_ring_prereduce(k, owners, L, Lnon=None, combine_extra=0.0):
    """Non-bit-exact: the owner columns run their own ring (lockstep, 1 unit per step), the non-owners pre-reduce chain
    by chain along their own sub-chain (each non-owner 1 unit per chain, hop L) and send the sum to the owner, whose
    combine adds it (combine_extra units). Returns the makespan."""
    n_own = len(owners)
    non = [c for c in range(k) if c not in owners]
    Lnon = L if Lnon is None else Lnon
    # owners: lockstep n_own steps.
    own_end = n_own
    # non-owners: column non[i] processes chains in a fixed order; sub-chain lag L per hop.
    t_sub = [0.0] * len(non)
    last = 0.0
    arr = {}
    for idx in range(n_own):
        prev = None
        for i in range(len(non)):
            s = max(t_sub[i], 0.0)
            e = s + 1
            if prev is not None:
                e = max(e, prev + L)
            t_sub[i] = e
            prev = e
        arr[idx] = prev + Lnon
        last = max(last, arr[idx])
    return max(own_end, last) + combine_extra


# ------------------------------------------------------- row-level DES model
class P:
    """Cycle costs (1350 MHz). Calibrated, see calibrate()."""

    mm_row = 330.0  # matmul, one tile-row of a 4x1 group, full K slice (2/3 of K live on y=5)
    mm_group = 60.0  # per group: init / acquire / commit
    pack_tile = 85.0  # fp32 L1-acc pack of one tile
    pack_call = 20.0  # per pack unit: reserve / push / STALLWAIT drain
    poll = 30.0  # a RISC noticing a counter change
    bw = 32.0  # bytes per cycle on one NoC0 link
    wlat = 150.0  # write issue -> landed + flush seen
    atomic = 100.0  # arrival semaphore increment latency
    tok = 30.0  # packer sees the token page freed
    inflight = 2
    tile_bytes = 4096


def ring_rows(c, k, owners, groups):
    """Column c's pack units in order: (j, owner, group index g, rows) list per step (HEAD order)."""
    seq = []
    for j in range(k):
        o = (c - 1 - j) % k
        if o not in owners:
            continue
        seq.append((j, o))
    return seq


def group_order(owner, sizes):
    """ring_group_row0: odd owners take the short (last) group first."""
    if len(sizes) > 1 and sizes[-1] != sizes[0] and owner & 1:
        return [len(sizes) - 1] + list(range(len(sizes) - 1))
    return list(range(len(sizes)))


def des(k, owners, rows, unit="group", p=P, mm_scale=None, out_n=1, sb_h=4, iters=60, pages=1):
    """Returns per-column packer end (cycles from walk start). unit: 'group' (HEAD) or 'row' (one token per tile-row)."""
    mm_scale = mm_scale or {}
    sizes = [min(sb_h, rows - g * sb_h) for g in range((rows + sb_h - 1) // sb_h)]
    row0 = [g * sb_h for g in range(len(sizes))]
    # arrival[(c, o, r0)] = time the unit starting at tile-row r0 of chain o landed (with its atomic) at column c
    arrival = {}
    ends = {}
    for _ in range(iters):
        changed = False
        for c in range(k):
            sc = mm_scale.get(c, 1.0)
            steps = ring_rows(c, k, owners, sizes)
            # flatten groups -> units
            groups = []
            for j, o in steps:
                for g in group_order(o, sizes):
                    groups.append((j, o, row0[g], sizes[g]))
            mfree = 0.0
            pfree = 0.0
            rfree = 0.0
            link = 0.0
            dest_rel = []  # packer done per group
            issued_land = []  # land times of issued writes (for inflight)
            pop_prev = 0.0  # reader popped the previous token
            seen_l = []  # reader saw each unit's push
            units = []
            for gi, (j, o, r0, nr) in enumerate(groups):
                tr = nr if unit == "group" else (1 if unit == "row" else int(unit))
                for r in range(0, nr, tr):
                    units.append((gi, j, o, r0 + r, min(tr, nr - r)))
            # math
            mend = []
            for gi, (j, o, r0, nr) in enumerate(groups):
                s = mfree
                if gi >= 2:
                    s = max(s, dest_rel[gi - 2]) if len(dest_rel) > gi - 2 else float("inf")
                e = s + p.mm_group + nr * p.mm_row * sc * out_n
                mend.append(e)
                mfree = e
                # packer for this group's units is computed below lazily; need dest_rel for gi-2 before math gi
                # -> interleave: compute this group's packs now (they only depend on mend[gi] and tokens)
                for ugi, uj, uo, ur0, unr in [u for u in units if u[0] == gi]:
                    has_in = uj >= 1
                    a_in = arrival.get((c, uo, ur0), float("inf")) if has_in else 0.0
                    ui = len(seen_l)
                    basis = pop_prev if pages == 1 else (seen_l[ui - pages] + 20 if ui >= pages else 0.0)
                    tok = max(basis, a_in + p.poll) + p.tok if has_in else basis + p.tok
                    ps = max(pfree, e, tok)
                    pe = ps + p.pack_call + unr * out_n * p.pack_tile
                    pfree = pe
                    seen = pe + p.poll
                    seen_l.append(seen)
                    send = uj + 1 < k
                    issue = max(seen, rfree)
                    if send:
                        if len(issued_land) >= p.inflight:
                            issue = max(issue, issued_land[-p.inflight] + p.poll)
                        nbytes = unr * out_n * p.tile_bytes
                        land = max(issue + p.wlat, link) + nbytes / p.bw
                        link = land
                        issued_land.append(land)
                        arr = land + p.atomic
                        key = ((c + 1) % k, uo, ur0)
                        if arrival.get(key) != arr:
                            arrival[key] = arr
                            changed = True
                        rfree = issue + 20
                    else:
                        rfree = issue
                    # the reader pops this token once the NEXT unit's incoming landed: modelled when that unit is seen
                    pop_prev = rfree
                dest_rel.append(pfree)
            # second pass for the pop dependency on the next unit's arrival: approximate by folding it into tok above
            if ends.get(c) != pfree:
                ends[c] = pfree
                changed = True
        if not changed:
            break
    return ends


def us(cyc):
    return cyc / 1350.0


def main():
    k, owners = 11, set(range(8))
    print("=== step model (L in steps; makespan in steps)")
    for L in (0.3, 0.5, 0.55):
        e = step_model(k, owners, L)
        print(
            f"HEAD 11 cols / 8 owners L={L}: max {max(e.values()):.2f}  per col "
            + " ".join(f"{e[c]:.2f}" for c in range(k))
        )
    print("lower bound (bit-exact fold o+1..o): col 0's first chain has passed cols 8,9,10 -> start >= 3L, end >= 8+3L")
    for L in (0.5, 0.55):
        print(
            f"  non-bit-exact owner ring + non-owner pre-reduce chain -> owner combine, L={L}: "
            f"{owner_ring_prereduce(k, owners, L):.2f} (+combine add)"
        )
    print("\n=== row-level DES, aspp.convs.3 grid row 3 (6 tile-rows: 4 + 2), col 10 = 2/3 K")
    sc = {10: 2.0 / 3.0}
    for unit, pg in (("group", 1), ("2", 1), ("row", 1), ("row", 2), ("row", 4)):
        e = des(k, owners, 6, unit=unit, mm_scale=sc, pages=pg)
        print(
            f"{unit:>5} pages {pg}: max {us(max(e.values())):6.2f} us  "
            + " ".join(f"{us(e[c]):5.2f}" for c in range(k))
        )
    print("rows 8 (4 + 4):")
    for unit, pg in (("group", 1), ("2", 1), ("row", 1), ("row", 2), ("row", 4)):
        e = des(k, owners, 8, unit=unit, mm_scale=sc, pages=pg)
        print(
            f"{unit:>5} pages {pg}: max {us(max(e.values())):6.2f} us  "
            + " ".join(f"{us(e[c]):5.2f}" for c in range(k))
        )
    print("\n=== all owners (res4.x: 8 cols, bm 7: 4 + 3)")

    class R4(P):
        mm_row = 170.0

    for unit, pg in (("group", 1), ("group", 2), ("2", 1), ("row", 1), ("row", 2), ("row", 4)):
        e = des(8, set(range(8)), 7, unit=unit, p=R4, pages=pg)
        print(
            f"{unit:>5} pages {pg}: max {us(max(e.values())):6.2f} us  "
            + " ".join(f"{us(e[c]):5.2f}" for c in range(8))
        )


def placements():
    """Owner placement (the candidate): which columns end a chain. Chains keep the physical +x ring; only the columns
    that sum (and store) an N-slice move. Not bit-identical by construction (each moved slice's fold starts elsewhere).
    """
    k, sc = 11, {10: 2.0 / 3.0}
    print("\n=== owner placement, 11 columns / 8 owners, group units (non-owning columns listed)")
    for idle in [(8, 9, 10), (3, 6, 10), (3, 7, 10), (2, 6, 10), (2, 5, 8), (0, 4, 8)]:
        own = set(range(k)) - set(idle)
        r = [us(max(des(k, own, rows, unit="group", mm_scale=sc).values())) for rows in (6, 8)]
        print(f"idle {idle}: rows 6 {r[0]:6.2f} us, rows 8 {r[1]:6.2f} us")


if __name__ == "__main__":
    main()
    placements()
