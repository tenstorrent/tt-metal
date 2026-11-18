# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin (-p asr_plugin, this dir on PYTHONPATH) for the t2p2_act_stream_rounds bake-off over the op's own perf
files, without touching them:
  ASR_SWAP=1      genify_matmul's WRITER kernel comes from ASR_KDIR (default ./kernels)
  ASR_DEFINES     '+'-joined defines appended to every kernel (e.g. GM_ASR_ACK_AHEAD)
Always prints one ASRPLAN line per op call: test id, the writer's act mode / ring depth / ahead / K-blocks / injectors
/ defines / source dir."""
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_current = {"id": "?"}


class _WriterSwap:
    def __init__(self, real):
        self.real = real

    def __truediv__(self, name):
        kdir = Path(os.environ.get("ASR_KDIR", str(HERE / "kernels")))
        return kdir / name if name == "genify_matmul_writer.cpp" else self.real / name


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _current["id"] = item.nodeid.split("::")[-1]


@pytest.fixture(autouse=True)
def _asr_swap(monkeypatch):
    from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd
    import importlib

    gm = importlib.import_module("ttnn.operations.genify_matmul.genify_matmul")

    if os.environ.get("ASR_SWAP") == "1":
        monkeypatch.setattr(pd, "KERNEL_DIR", _WriterSwap(pd.KERNEL_DIR))
    extra = [d for d in os.environ.get("ASR_DEFINES", "").split("+") if d]
    if os.environ.get("ASR_COUNTER") == "1":
        # The R1b A channel with the Counter data-ready signal (rounds counted, so cores may ack ahead).
        import ttnn

        def _attach_act_stream(descriptor, writer, core_ranges, device, injectors=1):
            config = ttnn.McastConfig(noc=pd.ACT_STREAM_NOC, handshake=True, data_ready=ttnn.McastDataReady.Counter)
            if injectors > 1:
                cores = ttnn.corerange_to_cores(core_ranges, None, True)
                picks = [cores[i * len(cores) // injectors] for i in range(injectors)]
                sender = ttnn.McastRotatingSenderConfig(
                    sender_cores=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in picks]),
                    sender_order=ttnn.McastCoreOrder.RowMajor,
                )
                writer.defines = list(writer.defines) + [("GM_ACT_INJECTORS", str(injectors))]
            else:
                sender = ttnn.McastFixedSenderConfig(sender_index=pd.ACT_STREAM_INJECTOR)
            mc = ttnn.Mcast(device, config, core_ranges, core_ranges.num_cores(), sender, ttnn.McastCoreOrder.RowMajor)
            mc.attach(descriptor, "act", [writer], 0)

        monkeypatch.setattr(pd, "_attach_act_stream", _attach_act_stream)
    if os.environ.get("ASR_HASH") == "1":
        # Bit-identity: hash every tensor the test reads back (CHECK=1 reads A, B, the output).
        import hashlib
        import ttnn

        real_to_torch = ttnn.to_torch

        def _to_torch(t, *a, **k):
            out = real_to_torch(t, *a, **k)
            h = hashlib.sha1(out.contiguous().view(-1).view(__import__("torch").uint8).numpy().tobytes()).hexdigest()
            print(f"ASRHASH {_current['id']} {tuple(out.shape)} {h[:16]}")
            return out

        monkeypatch.setattr(ttnn, "to_torch", _to_torch)
    orig = gm.create_program_descriptor

    def _traced(*args, **kwargs):
        desc = orig(*args, **kwargs)
        # Extra defines go on at descriptor time (the perf file's ABLATE_DEFINES knob is applied after fixtures).
        for k in desc.kernels:
            k.defines = list(k.defines) + [tuple(d.split("=", 1)) if "=" in d else (d, "1") for d in extra]
        try:
            for k in desc.kernels:
                if str(k.kernel_source).endswith("genify_matmul_writer.cpp"):
                    ct = list(k.compile_time_args)
                    defs = sorted(
                        {
                            f"{d[0]}={d[1]}" if d[1] != "1" else d[0]
                            for d in k.defines
                            if d[0].startswith(("GM_A", "GM_ACT", "GM_OUT"))
                        }
                    )
                    print(
                        f"ASRPLAN {_current['id']} act_mode={ct[8]} blk_m={ct[2]} blk_k={ct[11]} nkb={ct[12]} "
                        f"ring={ct[15]} ahead={ct[20]} ares={ct[19]} defs={'+'.join(defs)} "
                        f"src={Path(str(k.kernel_source)).parent.name}"
                    )
                    break
        except Exception as e:  # tracing only
            print(f"ASRPLAN {_current['id']} trace-failed {e}")
        return desc

    monkeypatch.setattr(gm, "create_program_descriptor", _traced)
