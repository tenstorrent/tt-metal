"""Host-only check of the overlapping cb_gamma page layouts: simulate the writer's byte writes per (dtype, page, gamma
layout, RM read scheme) and verify every page's row 0 of faces 0/1 holds the right gamma bytes, and that no in-flight
local copy reads a byte another in-flight copy writes (copies run concurrently between two barriers).
usage: python3 layout_check.py"""
import itertools

FACE = 16


def check(elem, P, tile_gamma, scheme, n=64):
    face_bytes = FACE * FACE * elem
    frow = FACE * elem
    mem = {}
    owner = lambda j, h: [("g", j, h, b) for b in range(frow)]  # gamma bytes of tile j half h

    issued = {}

    def write(addr, data, issue=True):
        for i, d in enumerate(data):
            if issue and addr + i in issued and issued[addr + i] != d:
                issued[addr + i] = "RACE"
            elif issue:
                issued[addr + i] = d
            mem[addr + i] = d

    for j in range(n):  # issue phase (all DRAM reads in flight together: no two may overlap with different data)
        t = P * j
        row = owner(j, 0) + owner(j, 1)  # the 32-datum DRAM row
        if tile_gamma:  # 2 reads of 64 B: [face0 row0 + rows below], [face1 row0 + rows below]
            write(t, owner(j, 0) + [("pad", j, 0, b) for b in range(64 - frow)] if frow < 64 else owner(j, 0))
            write(
                t + face_bytes, owner(j, 1) + [("pad", j, 1, b) for b in range(64 - frow)] if frow < 64 else owner(j, 1)
            )
        elif scheme == "direct2" and frow >= 64:
            write(t, owner(j, 0))
            write(t + face_bytes, owner(j, 1))
        else:
            write(t, row)
    if not tile_gamma and not (scheme == "direct2" and frow >= 64):
        srcs = [(P * j + frow, [mem.get(P * j + frow + i) for i in range(frow)]) for j in range(n)]
        dst_ranges = [range(P * j + face_bytes, P * j + face_bytes + frow) for j in range(n)]
        src_ranges = [range(P * j + frow, P * j + 2 * frow) for j in range(n)]
        allw = set(itertools.chain.from_iterable(dst_ranges))
        if any(a in allw for r in src_ranges for a in r):
            return "copy src/dst race"
        for j, (a, data) in enumerate(srcs):
            write(P * j + face_bytes, data, issue=False)
    need = set()
    for j in range(n):
        need |= set(range(P * j, P * j + frow)) | set(range(P * j + face_bytes, P * j + face_bytes + frow))
        if not tile_gamma and not (scheme == "direct2" and frow >= 64):
            need |= set(range(P * j + frow, P * j + 2 * frow))  # copy sources
    if any(issued.get(a) == "RACE" for a in need):
        return "in-flight DRAM read race on a needed byte"
    for j in range(n):
        t = P * j
        if [mem.get(t + i) for i in range(frow)] != owner(j, 0):
            return f"tile {j} face0 row0 wrong"
        if [mem.get(t + face_bytes + i) for i in range(frow)] != owner(j, 1):
            return f"tile {j} face1 row0 wrong"
    return "ok"


for elem, P in [(2, 192), (2, 1024), (2, 2048), (4, 192), (4, 384), (4, 2048)]:
    for tile_gamma in (True, False):
        for scheme in ["-"] if tile_gamma else ["direct", "direct2"]:
            print(
                f"elem={elem} P={P:5d} {'TILE' if tile_gamma else 'RM  '} {scheme:8s} -> {check(elem, P, tile_gamma, scheme)}"
            )
