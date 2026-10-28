import sys, ttnn

g = getattr(ttnn, "graph", None)
req = ("up_front_clear up_front_begin_collect up_front_end_collect up_front_num_unique up_front_compile").split()
missing = [a for a in req if not hasattr(g, a)]
sys.stdout.write("MISSING:" + ",".join(missing) if missing else "OK")
sys.exit(1 if missing else 0)
