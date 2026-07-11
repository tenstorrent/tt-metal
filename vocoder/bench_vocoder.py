# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark for the CodeHiFi-GAN vocoder forward (warm-cache, device-synced).

Toggle the DRAM width-slice optimization via SEAMLESS_VOCODER_CONV1D_DRAM_SLICE=1|0. See ../MEMORY.md.
"""

import os, time, statistics, torch, ttnn
from models.experimental.seamless_m4t_v2_large.reference.torch_code_hifigan import load_pretrained_code_hifigan
from models.experimental.seamless_m4t_v2_large.tests.pcc.pcc_test_common import weights_dir_or_skip
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import from_torch_uint32_rm, mesh_default_device
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_code_hifigan_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=32768, num_command_queues=2)

wdir = weights_dir_or_skip()
with mesh_default_device(dev):
    vocoder, cfg = load_pretrained_code_hifigan(wdir, dtype=torch.bfloat16)
    torch.manual_seed(0)
    unit_seq = 1024
    pad_id = int(cfg.t2u_pad_token_id)
    vocab = int(cfg.unit_hifi_gan_vocab_size)
    low = max(pad_id + 1, 2)
    high = max(low + 1, min(vocab - 1, low + 9999))
    input_ids = torch.randint(low, high, (1, unit_seq), dtype=torch.int64)
    speaker_id = torch.zeros(1, 1, dtype=torch.int64)
    lang_id = torch.zeros(1, 1, dtype=torch.int64)
    params = create_code_hifigan_parameters(vocoder, device=dev)
    tt_v = TTSeamlessM4Tv2CodeHifiGan(dev, params, cfg)
    ids = from_torch_uint32_rm(dev, input_ids)
    sp = from_torch_uint32_rm(dev, speaker_id)
    la = from_torch_uint32_rm(dev, lang_id)

    def timeit(n=5):
        # warm (compile)
        w, l = tt_v.forward(ids, sp, la)
        ttnn.synchronize_device(dev)
        ttnn.deallocate(w)
        ttnn.deallocate(l)
        ts = []
        for _ in range(n):
            t0 = time.time()
            w, l = tt_v.forward(ids, sp, la)
            ttnn.synchronize_device(dev)
            ts.append(time.time() - t0)
            ttnn.deallocate(w)
            ttnn.deallocate(l)
        return statistics.median(ts), min(ts)

    flag = os.environ.get("SEAMLESS_VOCODER_CONV1D_DRAM_SLICE", "1")
    med, mn = timeit()
    print(f"RESULT dram_slice={flag}: median={med*1000:.1f}ms min={mn*1000:.1f}ms")

ttnn.close_mesh_device(dev)
