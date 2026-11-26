"""Resolve the CT_* option defines of src/ kernels (unifdef-like, CT_ macros only).
usage: python3 resolve_defines.py <in.cpp> <out.cpp> CT_MR CT_FRAW   (listed = defined, other CT_* = undefined)"""
import re
import sys


def ev(expr, on):
    expr = re.sub(r"defined\s*\(\s*(CT_\w+)\s*\)", lambda m: "True" if m.group(1) in on else "False", expr)
    expr = expr.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
    return bool(eval(expr))


def resolve(lines, on):
    out, stack = [], []  # stack entries: [ours, emitting_parent, taken, active]
    for ln in lines:
        s = ln.strip()
        m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b\s*(.*)", s)
        active = all(e[3] for e in stack) if stack else True
        if m:
            d, arg = m.group(1), m.group(2).split("//")[0].strip()
            ours = "CT_" in arg if d in ("ifdef", "ifndef", "if", "elif") else None
            if d in ("ifdef", "ifndef", "if"):
                if ours:
                    c = (arg in on) if d == "ifdef" else (arg not in on) if d == "ifndef" else ev(arg, on)
                    stack.append([True, active, c, c])
                else:
                    stack.append([False, active, True, True])
                    if active:
                        out.append(ln)
                continue
            top = stack[-1]
            if d == "elif":
                if top[0]:
                    c = (not top[2]) and ev(arg, on)
                    top[3] = c
                    top[2] = top[2] or c
                elif top[1]:
                    out.append(ln)
                continue
            if d == "else":
                if top[0]:
                    top[3] = not top[2]
                    top[2] = True
                elif top[1]:
                    out.append(ln)
                continue
            if d == "endif":
                stack.pop()
                if not top[0] and top[1]:
                    out.append(ln)
                continue
        if active:
            out.append(ln)
    return out


if __name__ == "__main__":
    src, dst, *on = sys.argv[1:]
    open(dst, "w").writelines(resolve(open(src).readlines(), set(on)))
