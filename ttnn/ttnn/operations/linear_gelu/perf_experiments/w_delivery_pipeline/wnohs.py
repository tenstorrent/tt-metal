"""Host wrap: the op's descriptor with the FIRST McastConfig it builds (the W channel) switched to handshake=False +
Counter data-ready. Safe because cb_weight_resident is never popped (each chunk has its own slot) and the round-r
owner sends only after it has received rounds < r, so a Counter value >= r + 1 implies rounds 0..r have landed."""
import ttnn


def wrap(create):
    def create_wnohs(*args, **kwargs):
        real = ttnn.McastConfig
        calls = {"n": 0}

        def patched(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                cfg = real(**dict(kw, handshake=False))
                cfg.data_ready = ttnn.McastDataReady.Counter
                return cfg
            return real(**kw)

        ttnn.McastConfig = patched
        try:
            return create(*args, **kwargs)
        finally:
            ttnn.McastConfig = real

    return create_wnohs
