"""Host side of the W-no-handshake variant: the op's own descriptor with the FIRST McastConfig it builds (the W
channel) switched to handshake=False + Counter data-ready. The x channel keeps its handshake (ring slot reuse)."""
import ttnn


def wrap(create):
    def create_wnohs(*args, **kwargs):
        real = ttnn.McastConfig
        calls = {"n": 0}

        def patched(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                kw = dict(kw, handshake=False)
                cfg = real(**kw)
                cfg.data_ready = ttnn.McastDataReady.Counter
                return cfg
            return real(**kw)

        ttnn.McastConfig = patched
        try:
            return create(*args, **kwargs)
        finally:
            ttnn.McastConfig = real

    return create_wnohs
