"""pytest plugin: keep only items whose nodeid contains every '&'-separated substring of MMRS_NODE_FILTER, and
(optionally) the SHARD/NSHARDS slice of what remains (MMRS_SHARD=i/n) -- to run the golden suite in chunks."""
import os


def pytest_collection_modifyitems(config, items):
    f = os.environ.get("MMRS_NODE_FILTER")
    keep = [it for it in items if not f or all(s in it.nodeid for s in f.split("&"))]
    sh = os.environ.get("MMRS_SHARD")
    if sh:
        i, n = (int(x) for x in sh.split("/"))
        keep = keep[i::n]
    items[:] = keep
    print(f"[node_filter] {len(keep)} items")
