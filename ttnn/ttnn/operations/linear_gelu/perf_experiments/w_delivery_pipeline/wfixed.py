"""Host wrap: the W channel (the FIRST ttnn.Mcast the descriptor builds) uses ONE fixed sender per column, staggered
over the column rows (row (1 + column) % rows, the old chunk-0 owner). With the reader's lookahead the sender reads
chunk j + 1 while it multicasts chunk j: the sender is the only core that does not ingest the round's multicast, so
the DRAM read responses no longer share a receiver's ejection port with the multicast."""
import ttnn


def wrap(create, sender_index=1):
    def create_wfixed(*args, **kwargs):
        real = ttnn.Mcast
        calls = {"n": 0}

        def patched(device, cfg, cores, group_size, sender_cfg, order):
            calls["n"] += 1
            if calls["n"] == 1 and group_size > 1:
                sender_cfg = ttnn.McastFixedSenderConfig(
                    sender_index=sender_index, placement=ttnn.McastSenderPlacement.Staggered
                )
            return real(device, cfg, cores, group_size, sender_cfg, order)

        ttnn.Mcast = patched
        try:
            return create(*args, **kwargs)
        finally:
            ttnn.Mcast = real

    return create_wfixed
