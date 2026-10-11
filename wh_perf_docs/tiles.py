import re, collections


def tile_starts(svg, ncyc=60):
    """per packer: cycles where a tile starts (first DEST read of the tile), from the rect titles"""
    R = collections.defaultdict(dict)
    for c, p, b, r in re.findall(r"<title>cycle (\d+): packer (\d) reads DEST bank (\d)(, refused)?</title>", svg):
        R[int(p)][int(c)] = (int(b), bool(r))
    out = {}
    for p in range(4):
        cs = sorted(R[p])
        cand = [c for i, c in enumerate(cs) if R[p][c][0] == 0 and i and R[p][cs[i - 1]][0] == 3]
        anchor = min(cand, key=lambda c: abs(c - (11 + 2 * p)))
        st = [anchor]
        g = 0
        for c in cs:
            if c < st[-1]:
                continue
            if g >= 32 and R[p][c][0] == 0:
                st.append(c); g = 0
            if not R[p][c][1]:
                g += 1
        out[p] = st
    return out


if __name__ == "__main__":
    import json
    F = json.load(open("assets.json"))["figs"]
    for fi in (8, 9):
        print(fi, tile_starts(F[fi]))


GAP = 13  # extra space above each packer for its tile bar


def _sh(v):
    """vertical shift for an original y: GAP per packer band at or above it (bands start at 21 + 40 p)"""
    return GAP * sum(1 for p in range(4) if v >= 21 + 40 * p) if v >= 21 else 0


def _fix(m, attr):
    v = float(m.group(2))
    return f'{m.group(1)}="{v + _sh(v):g}"'


def add_tiles(svg, ncyc=60, x0=118.5, cw=14):
    """re-space the four packer bands and draw a tile bar above each packer's DEST row"""
    st = tile_starts(svg, ncyc)
    # rects: shift top and bottom separately, so boxes that span packers grow
    def rect(m):
        a = m.group(0)
        y = float(re.search(r' y="([\d.]+)"', a).group(1))
        h = re.search(r' height="([\d.]+)"', a)
        a = re.sub(r' y="[\d.]+"', f' y="{y + _sh(y):g}"', a, 1)
        if h:
            hv = float(h.group(1)); b = y + hv
            a = re.sub(r' height="[\d.]+"', f' height="{(b + _sh(b)) - (y + _sh(y)):g}"', a, 1)
        return a
    out = re.sub(r"<rect [^>]*>", rect, svg)
    out = re.sub(r'(<text [^>]*?) y="([\d.]+)"', lambda m: f'{m.group(1)} y="{float(m.group(2)) + _sh(float(m.group(2))):g}"', out)
    for at in ("y1", "y2", "cy"):
        out = re.sub(rf'( {at})="([\d.]+)"', lambda m: f'{m.group(1)}="{float(m.group(2)) + _sh(float(m.group(2))):g}"', out)
    vb = re.search(r'viewBox="0 0 (\d+) (\d+)" width="\d+" height="(\d+)"', out)
    W, H = int(vb.group(1)), int(vb.group(2)) + 4 * GAP
    out = out.replace(vb.group(0), f'viewBox="0 0 {W} {H}" width="{W}" height="{H}"', 1)
    bars = []
    for p in range(4):
        top = 25 + 40 * p + GAP * (p + 1) - GAP + 1  # just above the DEST row
        edges = [0] + st[p] + [ncyc]
        n0 = 1
        for i, (a, b) in enumerate(zip(edges, edges[1:])):
            if b <= a:
                continue
            k = n0 + i
            xa, xb = x0 + a * cw, x0 + b * cw - 1
            bars.append(f'<rect x="{xa:g}" y="{top}" width="{xb - xa:g}" height="10" rx="2" class="tbar t{k % 2}"><title>packer {p}: tile {k}, cycles {a}–{b - 1}</title></rect>')
            if b - a >= 5:
                bars.append(f'<text x="{xa + 3:g}" y="{top + 8}" class="tlab">tile {k}</text>')
            if a > 0:
                bars.append(f'<line x1="{xa - .5:g}" y1="{top}" x2="{xa - .5:g}" y2="{top + 10 + 14 + 22}" class="tedge"/>')
    return out.replace("</svg>", "".join(bars) + "</svg>")


TILE_CSS = """
.wave .tbar.t0{fill:var(--bk0)} .wave .tbar.t1{fill:var(--bk3)}
.wave .tlab{font:600 8px var(--mono);fill:var(--fg)}
.wave .tedge{stroke:var(--fg);stroke-width:1.4;stroke-dasharray:2 2}
"""
