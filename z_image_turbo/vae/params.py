"""
Weight loading and preparation for the VAE decoder TTNN model.

Loads PyTorch state_dict weights and converts them to the exact TTNN tensor
formats expected by the forward pass, replicating the consteval logic from
the generated main.py.
"""

import torch
import ttnn

from vae.consteval import (
    prepare_conv_weights,
    prepare_conv_bias,
    reshape_gn_weight,
    reshape_gn_weight_attn,
    make_scalar,
    make_ones_scalars,
    make_upsample_matrix,
)


# Arg index -> state_dict key mapping (from build_mapping.py analysis)
ARG_TO_STATE_DICT_KEY = {
    0: "conv_out.bias",
    1: "conv_out.weight",
    2: "conv_norm_out.bias",
    3: "conv_norm_out.weight",
    4: "up_blocks.3.resnets.2.conv2.bias",
    5: "up_blocks.3.resnets.2.conv2.weight",
    6: "up_blocks.3.resnets.2.norm2.bias",
    7: "up_blocks.3.resnets.2.norm2.weight",
    8: "up_blocks.3.resnets.2.conv1.bias",
    9: "up_blocks.3.resnets.2.conv1.weight",
    10: "up_blocks.3.resnets.2.norm1.bias",
    11: "up_blocks.3.resnets.2.norm1.weight",
    12: "up_blocks.3.resnets.1.conv2.bias",
    13: "up_blocks.3.resnets.1.conv2.weight",
    14: "up_blocks.3.resnets.1.norm2.bias",
    15: "up_blocks.3.resnets.1.norm2.weight",
    16: "up_blocks.3.resnets.1.conv1.bias",
    17: "up_blocks.3.resnets.1.conv1.weight",
    18: "up_blocks.3.resnets.1.norm1.bias",
    19: "up_blocks.3.resnets.1.norm1.weight",
    20: "up_blocks.3.resnets.0.conv2.bias",
    21: "up_blocks.3.resnets.0.conv2.weight",
    22: "up_blocks.3.resnets.0.norm2.bias",
    23: "up_blocks.3.resnets.0.norm2.weight",
    24: "up_blocks.3.resnets.0.conv1.bias",
    25: "up_blocks.3.resnets.0.conv1.weight",
    26: "up_blocks.3.resnets.0.norm1.bias",
    27: "up_blocks.3.resnets.0.norm1.weight",
    28: "up_blocks.2.upsamplers.0.conv.bias",
    29: "up_blocks.2.upsamplers.0.conv.weight",
    30: "up_blocks.2.resnets.2.conv2.bias",
    31: "up_blocks.2.resnets.2.conv2.weight",
    32: "up_blocks.2.resnets.2.norm2.bias",
    33: "up_blocks.2.resnets.2.norm2.weight",
    34: "up_blocks.2.resnets.2.conv1.bias",
    35: "up_blocks.2.resnets.2.conv1.weight",
    36: "up_blocks.2.resnets.2.norm1.bias",
    37: "up_blocks.2.resnets.2.norm1.weight",
    38: "up_blocks.2.resnets.1.conv2.bias",
    39: "up_blocks.2.resnets.1.conv2.weight",
    40: "up_blocks.2.resnets.1.norm2.bias",
    41: "up_blocks.2.resnets.1.norm2.weight",
    42: "up_blocks.2.resnets.1.conv1.bias",
    43: "up_blocks.2.resnets.1.conv1.weight",
    44: "up_blocks.2.resnets.1.norm1.bias",
    45: "up_blocks.2.resnets.1.norm1.weight",
    46: "up_blocks.2.resnets.0.conv2.bias",
    47: "up_blocks.2.resnets.0.conv2.weight",
    48: "up_blocks.2.resnets.0.norm2.bias",
    49: "up_blocks.2.resnets.0.norm2.weight",
    50: "up_blocks.2.resnets.0.conv1.bias",
    51: "up_blocks.2.resnets.0.conv1.weight",
    52: "up_blocks.2.resnets.0.norm1.bias",
    53: "up_blocks.2.resnets.0.norm1.weight",
    54: "up_blocks.1.upsamplers.0.conv.bias",
    55: "up_blocks.1.upsamplers.0.conv.weight",
    56: "up_blocks.1.resnets.2.conv2.bias",
    57: "up_blocks.1.resnets.2.conv2.weight",
    58: "up_blocks.1.resnets.2.norm2.bias",
    59: "up_blocks.1.resnets.2.norm2.weight",
    60: "up_blocks.1.resnets.2.conv1.bias",
    61: "up_blocks.1.resnets.2.conv1.weight",
    62: "up_blocks.1.resnets.2.norm1.bias",
    63: "up_blocks.1.resnets.2.norm1.weight",
    64: "up_blocks.1.resnets.1.conv2.bias",
    65: "up_blocks.1.resnets.1.conv2.weight",
    66: "up_blocks.1.resnets.1.norm2.bias",
    67: "up_blocks.1.resnets.1.norm2.weight",
    68: "up_blocks.1.resnets.1.conv1.bias",
    69: "up_blocks.1.resnets.1.conv1.weight",
    70: "up_blocks.1.resnets.1.norm1.bias",
    71: "up_blocks.1.resnets.1.norm1.weight",
    72: "up_blocks.1.resnets.0.conv2.bias",
    73: "up_blocks.1.resnets.0.conv2.weight",
    74: "up_blocks.1.resnets.0.norm2.bias",
    75: "up_blocks.1.resnets.0.norm2.weight",
    76: "up_blocks.1.resnets.0.conv1.bias",
    77: "up_blocks.1.resnets.0.conv1.weight",
    78: "up_blocks.1.resnets.0.norm1.bias",
    79: "up_blocks.1.resnets.0.norm1.weight",
    80: "up_blocks.0.upsamplers.0.conv.bias",
    81: "up_blocks.0.upsamplers.0.conv.weight",
    82: "up_blocks.0.resnets.2.conv2.bias",
    83: "up_blocks.0.resnets.2.conv2.weight",
    84: "up_blocks.0.resnets.2.norm2.bias",
    85: "up_blocks.0.resnets.2.norm2.weight",
    86: "up_blocks.0.resnets.2.conv1.bias",
    87: "up_blocks.0.resnets.2.conv1.weight",
    88: "up_blocks.0.resnets.2.norm1.bias",
    89: "up_blocks.0.resnets.2.norm1.weight",
    90: "up_blocks.0.resnets.1.conv2.bias",
    91: "up_blocks.0.resnets.1.conv2.weight",
    92: "up_blocks.0.resnets.1.norm2.bias",
    93: "up_blocks.0.resnets.1.norm2.weight",
    94: "up_blocks.0.resnets.1.conv1.bias",
    95: "up_blocks.0.resnets.1.conv1.weight",
    96: "up_blocks.0.resnets.1.norm1.bias",
    97: "up_blocks.0.resnets.1.norm1.weight",
    98: "up_blocks.0.resnets.0.conv2.bias",
    99: "up_blocks.0.resnets.0.conv2.weight",
    100: "up_blocks.0.resnets.0.norm2.bias",
    101: "up_blocks.0.resnets.0.norm2.weight",
    102: "up_blocks.0.resnets.0.conv1.bias",
    103: "up_blocks.0.resnets.0.conv1.weight",
    104: "up_blocks.0.resnets.0.norm1.bias",
    105: "up_blocks.0.resnets.0.norm1.weight",
    106: "mid_block.resnets.1.conv2.bias",
    107: "mid_block.resnets.1.conv2.weight",
    108: "mid_block.resnets.1.norm2.bias",
    109: "mid_block.resnets.1.norm2.weight",
    110: "mid_block.resnets.1.conv1.bias",
    111: "mid_block.resnets.1.conv1.weight",
    112: "mid_block.resnets.1.norm1.bias",
    113: "mid_block.resnets.1.norm1.weight",
    114: "mid_block.resnets.0.conv2.bias",
    115: "mid_block.resnets.0.conv2.weight",
    116: "mid_block.resnets.0.norm2.bias",
    117: "mid_block.resnets.0.norm2.weight",
    118: "mid_block.resnets.0.conv1.bias",
    119: "mid_block.resnets.0.conv1.weight",
    120: "mid_block.resnets.0.norm1.bias",
    121: "mid_block.resnets.0.norm1.weight",
    122: "conv_in.bias",
    123: "conv_in.weight",
    125: "mid_block.attentions.0.to_out.0.bias",
    126: "mid_block.attentions.0.to_out.0.weight",
    127: "mid_block.attentions.0.to_v.bias",
    128: "mid_block.attentions.0.to_v.weight",
    129: "mid_block.attentions.0.group_norm.bias",
    130: "mid_block.attentions.0.group_norm.weight",
    131: "mid_block.attentions.0.to_k.bias",
    132: "mid_block.attentions.0.to_k.weight",
    133: "mid_block.attentions.0.to_q.bias",
    134: "mid_block.attentions.0.to_q.weight",
    135: "up_blocks.2.resnets.0.conv_shortcut.bias",
    136: "up_blocks.2.resnets.0.conv_shortcut.weight",
    137: "up_blocks.3.resnets.0.conv_shortcut.bias",
    138: "up_blocks.3.resnets.0.conv_shortcut.weight",
}


def _to_bf16_rm(tensor_pt):
    """Convert a PyTorch tensor to a TTNN BF16 ROW_MAJOR host tensor."""
    return ttnn.from_torch(
        tensor_pt.to(torch.bfloat16),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
    )


def _to_bf16_tile_on_device(tensor_pt, device):
    """Convert a PyTorch tensor to a TTNN BF16 TILE tensor on device."""
    DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    t = ttnn.from_torch(
        tensor_pt.to(torch.bfloat16),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )
    return t


# ── Consteval-to-state-dict-key mapping ──────────────────────────────────────
# Extracted from consteval__main in main.py lines 10639-10924
# Format: consteval_idx -> (arg_idx, state_dict_key)
# This tells us which consteval function processes which weight.

# The consteval functions map (ce_idx -> arg_idx):
CE_TO_ARG = {
    0: 123,
    1: 8,
    2: 105,
    3: 54,
    4: 82,
    5: 97,
    6: 61,
    7: 76,
    8: 0,
    9: 43,
    10: 89,
    11: 90,
    12: 68,
    14: 117,
    15: 1,
    16: 28,
    17: 24,
    18: 113,
    19: 136,
    20: 9,
    21: 55,
    22: 106,
    23: 116,
    24: 104,
    25: 50,
    26: 138,
    27: 22,
    28: 15,
    29: 112,
    30: 111,
    31: 67,
    32: 91,
    33: 23,
    34: 16,
    35: 51,
    36: 98,
    37: 60,
    38: 135,
    39: 33,
    40: 119,
    41: 99,
    42: 84,
    43: 25,
    44: 44,
    45: 87,
    46: 18,
    47: 37,
    48: 110,
    49: 81,
    50: 40,
    51: 75,
    52: 103,
    53: 77,
    54: 30,
    55: 122,
    56: 29,
    57: 13,
    58: 96,
    59: 58,
    60: 94,
    61: 49,
    62: 56,
    63: 130,
    64: 42,
    65: 63,
    66: 108,
    67: 70,
    68: 35,
    69: 6,
    70: 53,
    71: 12,
    72: 101,
    73: 109,
    74: 27,
    75: 121,
    76: 46,
    77: 31,
    78: 65,
    79: 20,
    80: 114,
    82: 86,
    83: 72,
    84: 47,
    85: 11,
    86: 93,
    87: 4,
    88: 79,
    89: 32,
    90: 52,
    92: 3,
    93: 73,
    95: 80,
    96: 45,
    97: 92,
    98: 85,
    99: 39,
    100: 38,
    101: 137,
    102: 115,
    105: 74,
    107: 78,
    108: 59,
    109: 2,
    110: 107,
    111: 66,
    112: 41,
    113: 21,
    114: 10,
    115: 62,
    116: 17,
    117: 83,
    118: 14,
    120: 120,
    121: 7,
    122: 100,
    123: 102,
    125: 34,
    127: 71,
    128: 69,
    129: 5,
    130: 36,
    131: 64,
    132: 88,
    133: 129,
    134: 19,
    135: 48,
    136: 118,
    137: 95,
    138: 26,
    139: 57,
}


# ── Consteval function aliases (from the generated code) ─────────────────────
# format: alias_ce -> canonical_ce
CE_ALIASES = {
    5: 2,
    10: 2,
    11: 4,
    12: 7,
    14: 2,
    18: 2,
    22: 4,
    23: 2,
    24: 2,
    28: 27,
    29: 2,
    31: 2,
    32: 30,
    33: 27,
    34: 1,
    36: 4,
    37: 7,
    40: 30,
    41: 30,
    42: 2,
    44: 39,
    45: 30,
    46: 27,
    47: 39,
    48: 4,
    49: 6,
    50: 39,
    51: 2,
    52: 30,
    53: 6,
    57: 20,
    58: 2,
    59: 2,
    60: 4,
    61: 39,
    62: 7,
    64: 54,
    65: 2,
    66: 2,
    67: 2,
    68: 9,
    69: 27,
    70: 2,
    71: 1,
    72: 2,
    73: 2,
    74: 39,
    75: 2,
    76: 54,
    77: 9,
    78: 6,
    79: 1,
    80: 4,
    82: 4,
    83: 7,
    84: 9,
    85: 27,
    86: 2,
    87: 1,
    88: 2,
    89: 39,
    90: 2,
    92: 27,
    93: 6,
    95: 7,
    96: 39,
    97: 2,
    98: 2,
    99: 9,
    100: 54,
    102: 30,
    105: 2,
    107: 2,
    108: 2,
    109: 27,
    110: 30,
    111: 2,
    112: 39,
    113: 20,
    114: 27,
    115: 2,
    116: 20,
    117: 30,
    118: 27,
    120: 2,
    121: 27,
    122: 2,
    123: 4,
    125: 54,
    127: 2,
    128: 6,
    129: 20,
    130: 39,
    131: 7,
    132: 2,
    133: 63,
    134: 27,
    135: 39,
    136: 4,
    137: 30,
    138: 39,
    139: 6,
}


# ── Canonical consteval function types ───────────────────────────────────────
# These are the unique consteval function definitions (not aliases).
# Each maps to specific conv parameters.

# Conv weight configs: ce_idx -> (in_ch, out_ch, h, w, kernel, act_block_h)
CONV_WEIGHT_CONFIGS = {
    0: (16, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0),  # conv_in
    6: (512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 512->512 @ 128x128
    9: (256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 256->256 @ 256x256
    15: (128, 3, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 192),  # conv_out
    19: (512, 256, 256, 256, [1, 1], [1, 1], [0, 0, 0, 0], 0),  # conv_shortcut 512->256
    20: (128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 128->128 @ 512x512
    21: (512, 512, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 512->512 @ 256x256
    26: (256, 128, 512, 512, [1, 1], [1, 1], [0, 0, 0, 0], 0),  # conv_shortcut 256->128
    30: (512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0),  # 512->512 @ 64x64
    35: (512, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 512->256 @ 256x256
    43: (256, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 256->128 @ 512x512
    56: (256, 256, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 256->256 @ 512x512
}

# Conv bias configs: ce_idx -> (channels, in_ch, out_ch, h, w, kernel, act_block_h)
CONV_BIAS_CONFIGS = {
    1: (128, 128, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 128ch bias @ 512x512
    3: (512, 512, 512, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 512ch bias @ 256x256 DRAM
    4: (512, 512, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0),  # 512ch bias @ 64x64
    7: (512, 512, 512, 128, 128, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 512ch bias @ 128x128
    8: (3, 128, 3, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 192),  # 3ch bias (conv_out)
    16: (256, 256, 256, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 256ch bias @ 512x512
    17: (128, 256, 128, 512, 512, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 128ch bias, in=256 @ 512
    25: (256, 512, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 256ch bias, in=512 @ 256
    38: (256, 512, 256, 256, 256, [1, 1], [1, 1], [0, 0, 0, 0], 0),  # conv_shortcut bias
    54: (256, 256, 256, 256, 256, [3, 3], [1, 1], [1, 1, 1, 1], 1024),  # 256ch bias @ 256x256
    55: (512, 16, 512, 64, 64, [3, 3], [1, 1], [1, 1, 1, 1], 0),  # conv_in bias
    101: (128, 256, 128, 512, 512, [1, 1], [1, 1], [0, 0, 0, 0], 0),  # conv_shortcut bias 256->128
}

# GN weight configs: ce_idx -> channels
GN_WEIGHT_CONFIGS = {
    2: 512,  # standard GN for 512ch
    27: 128,  # standard GN for 128ch
    39: 256,  # standard GN for 256ch
    63: 512,  # attention GN for 512ch (different permute pattern)
}


def load_weights(state_dict, device):
    """Load all VAE decoder weights and produce processed TTNN tensors.

    This replicates the consteval functions and load_inputs from the generated code.

    Args:
        state_dict: PyTorch state_dict from the VAE decoder
        device: TTNN mesh device

    Returns:
        dict mapping internal weight names to processed TTNN tensors.
        The keys follow the consteval cache key naming: "main_const_eval_N"
    """
    DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

    # Build arg_idx -> raw TTNN host tensor
    raw_tensors = {}
    for arg_idx, sd_key in ARG_TO_STATE_DICT_KEY.items():
        raw_tensors[arg_idx] = _to_bf16_rm(state_dict[sd_key])

    # Process each consteval entry
    ce_cache = {}

    for ce_idx in range(140):
        # Skip no-input constevals (scalars and upsample matrices) handled separately
        if ce_idx in (13, 81, 91, 94, 103, 104, 106, 119, 124, 126):
            continue

        # Get the canonical function index
        canon = CE_ALIASES.get(ce_idx, ce_idx)

        # Get the arg index
        if ce_idx not in CE_TO_ARG:
            continue
        arg_idx = CE_TO_ARG[ce_idx]
        raw = raw_tensors[arg_idx]

        # Dispatch based on canonical function type
        if canon in CONV_WEIGHT_CONFIGS:
            cfg = CONV_WEIGHT_CONFIGS[canon]
            ce_cache[f"main_const_eval_{ce_idx}"] = [
                prepare_conv_weights(
                    raw,
                    device,
                    in_channels=cfg[0],
                    out_channels=cfg[1],
                    batch_size=1,
                    input_height=cfg[2],
                    input_width=cfg[3],
                    kernel_size=cfg[4],
                    stride=cfg[5],
                    padding=cfg[6],
                    dilation=[1, 1],
                    groups=1,
                    act_block_h_override=cfg[7],
                )
            ]
        elif canon in CONV_BIAS_CONFIGS:
            cfg = CONV_BIAS_CONFIGS[canon]
            ce_cache[f"main_const_eval_{ce_idx}"] = [
                prepare_conv_bias(
                    raw,
                    device,
                    channels=cfg[0],
                    in_channels=cfg[1],
                    out_channels=cfg[2],
                    batch_size=1,
                    input_height=cfg[3],
                    input_width=cfg[4],
                    kernel_size=cfg[5],
                    stride=cfg[6],
                    padding=cfg[7],
                    dilation=[1, 1],
                    groups=1,
                    act_block_h_override=cfg[8],
                )
            ]
        elif canon in GN_WEIGHT_CONFIGS:
            channels = GN_WEIGHT_CONFIGS[canon]
            if canon == 63:
                ce_cache[f"main_const_eval_{ce_idx}"] = [reshape_gn_weight_attn(raw, device, channels)]
            else:
                ce_cache[f"main_const_eval_{ce_idx}"] = [reshape_gn_weight(raw, device, channels)]
        else:
            raise ValueError(f"Unknown canonical consteval function {canon} for ce_idx={ce_idx}")

    # ── No-input constevals (scalars and upsample matrices) ──────────────────

    # var_1: epsilon-like scalar for 256ch @ 256x256 GN
    ce_cache["main_const_eval_13"] = [make_scalar(device, 1.9073486328125e-06)]

    # var_3, var_4: ones scalars for divide-by-1.0
    var_3, var_4 = make_ones_scalars(device)
    ce_cache["main_const_eval_81"] = [var_3, var_4]

    # var_5: GN epsilon
    ce_cache["main_const_eval_91"] = [make_scalar(device, 9.9999999747524271e-07)]

    # var_6: 1/(8*262144) = 1/2097152 for 256ch @ 512x512
    ce_cache["main_const_eval_94"] = [make_scalar(device, 4.76837158203125e-07)]

    # var_7: 1/(16*16384) = 1/262144 for 512ch @ 128x128
    ce_cache["main_const_eval_103"] = [make_scalar(device, 3.814697265625e-06)]

    # var_8: upsample matrix 256->512 (for 64->128 upsample, output side)
    ce_cache["main_const_eval_104"] = [make_upsample_matrix(device, 256, 512)]

    # var_9: 1/(16*4096) = 1/65536 for 512ch @ 64x64
    ce_cache["main_const_eval_106"] = [make_scalar(device, 1.52587890625e-05)]

    # var_10: upsample matrix 128->256
    ce_cache["main_const_eval_119"] = [make_upsample_matrix(device, 128, 256)]

    # var_11: 1/(16*65536) = 1/1048576 for 512ch @ 256x256 (or 128ch @ 512x512)
    ce_cache["main_const_eval_124"] = [make_scalar(device, 9.5367431640625e-07)]

    # var_12: upsample matrix 64->128
    ce_cache["main_const_eval_126"] = [make_upsample_matrix(device, 64, 128)]

    # ── Attention weights (loaded directly to device, not via consteval) ─────
    attn_args = {}
    for arg_idx in (125, 126, 127, 128, 131, 132, 133, 134):
        sd_key = ARG_TO_STATE_DICT_KEY[arg_idx]
        attn_args[arg_idx] = _to_bf16_tile_on_device(state_dict[sd_key], device)

    return ce_cache, attn_args
