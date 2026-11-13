import ttnn
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache

c = _FabricMeshCache()
m = c.get((1, 4))
node = lambda i: m.get_fabric_node_id(ttnn.MeshCoordinate(0, i))
for s in range(4):
    print("chip", s, "node", node(s))
    for d in range(4):
        if d == s:
            continue
        try:
            print(
                "  ->",
                d,
                ttnn.get_eth_forwarding_direction(node(s), node(d)),
                "links",
                list(ttnn.get_forwarding_link_indices(node(s), node(d))),
            )
        except Exception as e:
            print("  ->", d, "ERR", e)
c.close()
