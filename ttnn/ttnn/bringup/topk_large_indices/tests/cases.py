# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One case per distinct ttnn.bringup.topk_large_indices call a model makes (bringup-fork-tests skill). Append only;
never edit or loosen another model's case. Shapes are per device, exactly as captured (fork_calls.json)."""

_DS41 = {
    "model": "deepseek_v41_flash_d_p",
    "task": "DS41-151",
    "mesh": [2, 2],
    "device_params": {"fabric_config": "FABRIC_2D", "l1_small_size": 24576},
    "neginf_from": None,
    "seed": 0,
}

CASES = [
    {
        # deepseek_v41_flash_d_p DS41-151 (scaled 2x2, ladder rung last; tt/indexer.py TtIndexer.topk_of): the index
        # top-k 512 of the chip's 512 query rows over the compressed keys, the ratio-2 layers at the last chunk (55296 / 2).
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-index-k512-27648",
        "sig": "a49d0dc1d0",
        "input": {"shape": [1, 1, 512, 27648], "dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "k": 512,
        "valid_length": 27648,
    },
    {
        # the same at ratio 1 (the decoder layers dec_candidate / dec_reindex, 55296 keys)
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-index-k512-55296",
        "sig": "51aed81c27",
        "input": {"shape": [1, 1, 512, 55296], "dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "k": 512,
        "valid_length": 55296,
    },
    {
        # tt/indexer.py TtCandidateIndexer.candidates_of: the candidate blocks' top 2048 of 6912 block scores a row
        **_DS41,
        "id": "deepseek_v41_flash_d_p-2x2-candidate-k2048-6912",
        "sig": "d040f5a6f7",
        "input": {"shape": [1, 1, 512, 6912], "dtype": "BFLOAT16", "layout": "ROW_MAJOR"},
        "k": 2048,
        "valid_length": 6912,
    },
]
