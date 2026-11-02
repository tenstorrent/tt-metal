"""Minimal partial unifdef: resolve #if/#ifdef/#elif/#else/#endif whose value is fixed by KNOWN; keep the rest."""
import itertools, re, sys

KNOWN = {"FLAT_CMP": True, "FLAT_NO_SQ_AHEAD": False}


def value(expr):
    expr = expr.strip()
    syms = sorted(set(re.findall(r"defined\s*\(\s*(\w+)\s*\)", expr)))
    unknown = [s for s in syms if s not in KNOWN]
    results = set()
    for combo in itertools.product([False, True], repeat=len(unknown)):
        env = dict(KNOWN, **dict(zip(unknown, combo)))
        e = re.sub(r"defined\s*\(\s*(\w+)\s*\)", lambda m: str(env[m.group(1)]), expr)
        e = e.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
        results.add(bool(eval(e)))
    return results.pop() if len(results) == 1 else None


def norm(line):
    s = line.strip()
    m = re.match(r"#\s*ifdef\s+(\w+)", s)
    if m:
        return "if", f"defined({m.group(1)})"
    m = re.match(r"#\s*ifndef\s+(\w+)", s)
    if m:
        return "if", f"!defined({m.group(1)})"
    m = re.match(r"#\s*if\s+(.*)", s)
    if m:
        return "if", m.group(1)
    m = re.match(r"#\s*elif\s+(.*)", s)
    if m:
        return "elif", m.group(1)
    if re.match(r"#\s*else\b", s):
        return "else", None
    if re.match(r"#\s*endif\b", s):
        return "endif", None
    return None, None


def process(lines):
    out = []
    # stack of frames: {resolved: bool, emitting_parent, taken (bool), mode: 'strip'|'keep', cur_active}
    stack = []

    def active():
        return all(f["cur"] for f in stack)

    for line in lines:
        kind, expr = norm(line)
        if kind == "if":
            v = value(expr) if ("defined" in expr) else None
            if v is None:
                stack.append({"keep": True, "cur": True, "taken": False, "kept_any": True})
                if active():
                    out.append(line)
            else:
                stack.append({"keep": False, "cur": v, "taken": v})
            continue
        if kind == "elif":
            f = stack[-1]
            if f["keep"]:
                if all(g["cur"] for g in stack[:-1]):
                    out.append(line)
                continue
            v = False if f["taken"] else value(expr)
            if v is None:
                raise SystemExit("unresolvable elif after resolved if: " + line)
            f["cur"] = (not f["taken"]) and v
            f["taken"] = f["taken"] or v
            continue
        if kind == "else":
            f = stack[-1]
            if f["keep"]:
                if all(g["cur"] for g in stack[:-1]):
                    out.append(line)
                continue
            f["cur"] = not f["taken"]
            f["taken"] = True
            continue
        if kind == "endif":
            f = stack.pop()
            if f["keep"] and active():
                out.append(line)
            continue
        if active():
            out.append(line)
    return out


src, dst = sys.argv[1], sys.argv[2]
open(dst, "w").write("".join(process(open(src).readlines())))
