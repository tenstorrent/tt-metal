"""Host wrap: the W channel (the FIRST ttnn.Mcast the descriptor builds) rotates only between the two END cores of
each column (top and bottom row; left and right column when transposed). A NoC0 multicast from a sender inside its
column rectangle wraps around the torus and crosses the links below the sender twice, halving its bandwidth; from
either end every link is crossed once."""
import ttnn


def wrap(create):
    def create_wends(*args, **kwargs):
        real = ttnn.Mcast
        calls = {"n": 0}

        def patched(device, cfg, cores, group_size, sender_cfg, order):
            calls["n"] += 1
            if calls["n"] == 1 and group_size > 2:
                box = cores.bounding_box()
                lo, hi = box.start, box.end
                C = ttnn.CoreCoord
                if order == ttnn.McastCoreOrder.ColumnMajor:  # groups are physical columns
                    ends = [ttnn.CoreRange(C(lo.x, lo.y), C(hi.x, lo.y)), ttnn.CoreRange(C(lo.x, hi.y), C(hi.x, hi.y))]
                else:  # groups are physical rows
                    ends = [ttnn.CoreRange(C(lo.x, lo.y), C(lo.x, hi.y)), ttnn.CoreRange(C(hi.x, lo.y), C(hi.x, hi.y))]
                sender_cfg = ttnn.McastRotatingSenderConfig(sender_cores=ttnn.CoreRangeSet(ends))
            return real(device, cfg, cores, group_size, sender_cfg, order)

        ttnn.Mcast = patched
        try:
            return create(*args, **kwargs)
        finally:
            ttnn.Mcast = real

    return create_wends
