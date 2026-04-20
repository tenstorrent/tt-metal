# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Consteval functions for ZImageTransformer TTNN graph.

Replaces 517 auto-generated main_const_eval_N() functions with 7 transformation
patterns applied via a data-driven lookup table.

Transformation types:
  B  (basic, 238):          to_device + to_layout(TILE)
  F  (typecast_f32, 37):    B + typecast(FLOAT32)          -- bias vectors (adaLN, t_embedder)
  P  (permute_typecast, 37):B + permute([1,0]) + typecast   -- adaLN weight matrices
  R0 (reshape_64_2, 68):    B + reshape([1,1,1,64,2])       -- norm_q / norm_k weights
  R1 (reshape_1_1_D, 68):   B + reshape([1,1,3840])         -- attention_norm2 / ffn_norm2
  R2 (reshape_1_D, 68):     B + reshape([1,3840])           -- attention_norm1 / ffn_norm1
  R3 (reshape_cap, 1):      B + reshape([1,2560])           -- cap_embedder.0.weight
"""

import math

import torch
import ttnn

DRAM_MC = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

# ---------------------------------------------------------------------------
# Consteval lookup table: ce_idx -> (arg_idx, transform_type)
#
# ce_idx   : key used by _main() as _cached__main["main_const_eval_{ce_idx}"]
# arg_idx  : index into the 529-element inputs array
# type     : one of B / F / P / R0 / R1 / R2 / R3
# ---------------------------------------------------------------------------
CONSTEVAL_MAP = {
    0: (365, "F"),
    1: (271, "B"),
    2: (186, "B"),
    3: (486, "R0"),
    4: (3, "P"),
    5: (500, "B"),
    6: (326, "P"),
    7: (240, "B"),
    8: (472, "B"),
    9: (387, "B"),
    10: (378, "B"),
    11: (124, "R2"),
    12: (158, "R2"),
    13: (514, "R0"),
    14: (383, "B"),
    15: (401, "R0"),
    16: (155, "R1"),
    17: (182, "F"),
    18: (39, "R1"),
    19: (212, "F"),
    20: (415, "B"),
    21: (381, "R0"),
    22: (143, "P"),
    23: (172, "F"),
    24: (16, "B"),
    25: (395, "B"),
    26: (409, "R0"),
    27: (280, "B"),
    28: (177, "B"),
    29: (13, "P"),
    30: (217, "B"),
    31: (249, "R1"),
    32: (429, "R0"),
    33: (209, "R1"),
    34: (79, "R1"),
    35: (200, "B"),
    36: (115, "R1"),
    37: (257, "B"),
    38: (407, "B"),
    39: (403, "B"),
    40: (102, "F"),
    41: (198, "R2"),
    42: (370, "B"),
    43: (118, "R2"),
    44: (87, "B"),
    45: (357, "B"),
    46: (195, "R1"),
    47: (323, "B"),
    48: (468, "B"),
    49: (204, "R2"),
    50: (376, "R0"),
    51: (494, "R0"),
    52: (460, "B"),
    53: (512, "B"),
    54: (302, "F"),
    55: (254, "R2"),
    56: (70, "B"),
    57: (141, "B"),
    58: (508, "B"),
    59: (506, "R0"),
    60: (314, "B"),
    61: (248, "R2"),
    62: (466, "R0"),
    63: (317, "R1"),
    64: (61, "B"),
    65: (65, "R1"),
    66: (44, "R2"),
    67: (307, "B"),
    68: (276, "B"),
    69: (432, "B"),
    70: (160, "B"),
    71: (528, "B"),
    72: (150, "B"),
    73: (292, "F"),
    74: (455, "B"),
    75: (454, "R0"),
    76: (299, "R1"),
    77: (38, "R2"),
    78: (289, "R1"),
    79: (151, "B"),
    80: (306, "B"),
    81: (339, "B"),
    82: (449, "R0"),
    83: (513, "B"),
    84: (52, "F"),
    85: (120, "B"),
    86: (517, "B"),
    87: (352, "B"),
    89: (297, "B"),
    90: (74, "R2"),
    91: (132, "F"),
    92: (34, "R2"),
    93: (389, "R0"),
    94: (137, "B"),
    95: (129, "R1"),
    97: (169, "R1"),
    98: (469, "R0"),
    99: (28, "R2"),
    100: (288, "R2"),
    101: (452, "B"),
    102: (68, "R2"),
    103: (511, "R0"),
    104: (520, "B"),
    105: (25, "R1"),
    106: (311, "B"),
    107: (190, "B"),
    108: (341, "B"),
    109: (146, "B"),
    110: (56, "B"),
    111: (313, "R1"),
    112: (267, "B"),
    113: (434, "R0"),
    114: (463, "B"),
    115: (43, "P"),
    116: (303, "P"),
    117: (259, "R1"),
    118: (51, "B"),
    119: (461, "R0"),
    120: (47, "B"),
    121: (440, "B"),
    122: (164, "R2"),
    123: (316, "R2"),
    124: (30, "B"),
    125: (262, "F"),
    126: (467, "B"),
    127: (294, "R2"),
    128: (273, "P"),
    129: (526, "R0"),
    130: (448, "B"),
    131: (285, "R1"),
    132: (181, "B"),
    133: (21, "B"),
    134: (446, "R0"),
    135: (509, "R0"),
    136: (134, "R2"),
    137: (414, "R0"),
    138: (148, "R2"),
    139: (483, "B"),
    140: (250, "B"),
    141: (377, "B"),
    142: (183, "P"),
    143: (60, "B"),
    144: (350, "R2"),
    145: (481, "R0"),
    146: (491, "R0"),
    147: (505, "B"),
    148: (91, "B"),
    149: (239, "R1"),
    150: (398, "B"),
    151: (404, "R0"),
    152: (493, "B"),
    153: (40, "B"),
    154: (154, "R2"),
    155: (373, "B"),
    156: (390, "B"),
    157: (410, "B"),
    159: (133, "P"),
    160: (281, "B"),
    161: (241, "B"),
    162: (49, "R1"),
    163: (396, "R0"),
    164: (145, "R1"),
    165: (57, "B"),
    166: (380, "B"),
    167: (221, "B"),
    168: (290, "B"),
    169: (475, "B"),
    170: (243, "P"),
    171: (497, "B"),
    172: (457, "B"),
    173: (284, "R2"),
    174: (100, "B"),
    175: (515, "B"),
    176: (278, "R2"),
    177: (247, "B"),
    178: (453, "B"),
    179: (111, "B"),
    180: (384, "R0"),
    181: (31, "B"),
    182: (487, "B"),
    183: (392, "B"),
    184: (485, "B"),
    185: (521, "R0"),
    186: (310, "B"),
    187: (301, "B"),
    188: (230, "B"),
    189: (420, "B"),
    190: (479, "R0"),
    191: (88, "R2"),
    192: (386, "R0"),
    193: (428, "B"),
    194: (94, "R2"),
    195: (388, "B"),
    196: (109, "R1"),
    197: (85, "R1"),
    198: (426, "R0"),
    199: (117, "B"),
    200: (450, "B"),
    201: (427, "B"),
    202: (522, "B"),
    203: (73, "P"),
    204: (523, "B"),
    205: (62, "F"),
    206: (6, "F"),
    207: (266, "B"),
    209: (445, "B"),
    210: (127, "B"),
    211: (75, "R1"),
    212: (171, "B"),
    213: (24, "R2"),
    214: (119, "R1"),
    215: (263, "P"),
    216: (275, "R1"),
    217: (256, "B"),
    218: (451, "R0"),
    220: (268, "R2"),
    221: (123, "P"),
    222: (170, "B"),
    223: (518, "B"),
    224: (66, "B"),
    225: (252, "F"),
    226: (516, "R0"),
    227: (265, "R1"),
    228: (161, "B"),
    229: (253, "P"),
    231: (456, "R0"),
    232: (53, "P"),
    233: (335, "R0"),
    234: (84, "R2"),
    235: (15, "R1"),
    236: (274, "R2"),
    237: (110, "B"),
    238: (385, "B"),
    239: (136, "B"),
    240: (397, "B"),
    241: (231, "B"),
    242: (444, "R0"),
    243: (179, "R1"),
    244: (325, "F"),
    245: (78, "R2"),
    246: (421, "R0"),
    247: (18, "R2"),
    248: (474, "R0"),
    249: (379, "R0"),
    250: (187, "B"),
    251: (291, "B"),
    252: (423, "B"),
    253: (192, "F"),
    254: (318, "B"),
    255: (488, "B"),
    256: (214, "R2"),
    257: (2, "F"),
    258: (492, "B"),
    259: (480, "B"),
    260: (315, "B"),
    261: (196, "B"),
    262: (208, "R2"),
    263: (300, "B"),
    264: (101, "B"),
    265: (309, "R1"),
    266: (205, "R1"),
    267: (360, "R2"),
    268: (478, "B"),
    269: (162, "F"),
    270: (50, "B"),
    271: (312, "R2"),
    272: (405, "B"),
    273: (193, "P"),
    274: (504, "R0"),
    275: (144, "R2"),
    276: (19, "R1"),
    277: (343, "B"),
    278: (138, "R2"),
    279: (391, "R0"),
    280: (393, "B"),
    281: (27, "B"),
    282: (175, "R1"),
    283: (308, "R2"),
    284: (59, "R1"),
    285: (510, "B"),
    286: (524, "R0"),
    287: (462, "B"),
    288: (358, "F"),
    289: (482, "B"),
    290: (90, "B"),
    291: (371, "R0"),
    292: (498, "B"),
    293: (351, "R1"),
    294: (490, "B"),
    295: (496, "R0"),
    296: (305, "R1"),
    297: (354, "R2"),
    298: (121, "B"),
    299: (81, "B"),
    300: (476, "R0"),
    301: (41, "B"),
    302: (439, "R0"),
    303: (458, "B"),
    304: (122, "F"),
    305: (502, "B"),
    306: (327, "R3"),
    307: (346, "B"),
    308: (203, "P"),
    309: (63, "P"),
    310: (338, "B"),
    311: (413, "B"),
    312: (67, "B"),
    313: (321, "R1"),
    314: (411, "R0"),
    315: (215, "R1"),
    316: (135, "R1"),
    317: (14, "R2"),
    318: (277, "B"),
    319: (251, "B"),
    320: (112, "F"),
    321: (374, "R0"),
    322: (180, "B"),
    323: (206, "B"),
    324: (76, "B"),
    325: (126, "B"),
    326: (527, "B"),
    327: (260, "B"),
    328: (197, "B"),
    329: (399, "R0"),
    330: (269, "R1"),
    331: (189, "R1"),
    332: (433, "B"),
    333: (255, "R1"),
    334: (464, "R0"),
    335: (369, "R0"),
    336: (234, "R2"),
    337: (224, "R2"),
    338: (86, "B"),
    339: (83, "P"),
    340: (340, "R0"),
    341: (139, "R1"),
    342: (55, "R1"),
    343: (364, "R2"),
    344: (103, "P"),
    345: (58, "R2"),
    346: (242, "F"),
    347: (319, "B"),
    348: (107, "B"),
    349: (431, "R0"),
    350: (64, "R2"),
    351: (336, "B"),
    352: (417, "B"),
    353: (218, "R2"),
    354: (258, "R2"),
    355: (437, "B"),
    356: (130, "B"),
    357: (228, "R2"),
    358: (422, "B"),
    359: (194, "R2"),
    360: (470, "B"),
    361: (418, "B"),
    362: (225, "R1"),
    363: (416, "R0"),
    364: (213, "P"),
    365: (98, "R2"),
    366: (484, "R0"),
    368: (402, "B"),
    369: (425, "B"),
    370: (113, "P"),
    371: (233, "P"),
    372: (264, "R2"),
    373: (246, "B"),
    374: (211, "B"),
    375: (104, "R2"),
    376: (99, "R1"),
    377: (95, "R1"),
    378: (7, "P"),
    379: (356, "B"),
    380: (220, "B"),
    381: (324, "R2"),
    382: (237, "B"),
    383: (72, "F"),
    384: (499, "R0"),
    385: (116, "B"),
    386: (125, "R1"),
    387: (347, "B"),
    388: (69, "R1"),
    389: (419, "R0"),
    390: (375, "B"),
    391: (128, "R2"),
    392: (229, "R1"),
    393: (77, "B"),
    394: (29, "R1"),
    395: (207, "B"),
    396: (382, "B"),
    397: (495, "B"),
    398: (176, "B"),
    399: (322, "B"),
    400: (400, "B"),
    401: (477, "B"),
    402: (71, "B"),
    403: (202, "F"),
    404: (473, "B"),
    405: (227, "B"),
    406: (359, "P"),
    407: (424, "R0"),
    409: (199, "R1"),
    410: (219, "R1"),
    411: (80, "B"),
    412: (163, "P"),
    413: (114, "R2"),
    414: (361, "R1"),
    415: (9, "R1"),
    416: (216, "B"),
    417: (26, "B"),
    418: (304, "R2"),
    419: (12, "F"),
    420: (342, "R0"),
    421: (173, "P"),
    422: (17, "B"),
    423: (188, "R2"),
    424: (108, "R2"),
    425: (344, "B"),
    426: (185, "R1"),
    427: (142, "F"),
    428: (46, "B"),
    429: (22, "F"),
    430: (167, "B"),
    431: (430, "B"),
    432: (5, "P"),
    433: (236, "B"),
    434: (348, "F"),
    435: (447, "B"),
    436: (210, "B"),
    437: (298, "R2"),
    438: (20, "B"),
    439: (174, "R2"),
    440: (35, "R1"),
    441: (295, "R1"),
    442: (525, "B"),
    443: (147, "B"),
    444: (159, "R1"),
    445: (283, "P"),
    446: (168, "R2"),
    447: (366, "P"),
    448: (232, "F"),
    449: (443, "B"),
    450: (156, "B"),
    451: (42, "F"),
    452: (519, "R0"),
    453: (223, "P"),
    455: (32, "F"),
    456: (153, "P"),
    457: (363, "B"),
    458: (165, "R1"),
    459: (245, "R1"),
    460: (37, "B"),
    461: (92, "F"),
    462: (355, "R1"),
    463: (238, "R2"),
    464: (286, "B"),
    465: (459, "R0"),
    466: (96, "B"),
    467: (282, "F"),
    468: (272, "F"),
    469: (191, "B"),
    470: (345, "R1"),
    471: (82, "F"),
    472: (441, "R0"),
    473: (353, "B"),
    474: (235, "R1"),
    475: (201, "B"),
    476: (93, "P"),
    477: (11, "B"),
    478: (287, "B"),
    479: (435, "B"),
    480: (337, "R0"),
    481: (442, "B"),
    482: (106, "B"),
    483: (293, "P"),
    484: (89, "R1"),
    485: (296, "B"),
    486: (438, "B"),
    487: (105, "R1"),
    488: (97, "B"),
    489: (436, "R0"),
    490: (184, "R2"),
    491: (320, "R2"),
    492: (408, "B"),
    493: (131, "B"),
    494: (503, "B"),
    495: (489, "R0"),
    496: (362, "B"),
    497: (226, "B"),
    498: (33, "P"),
    499: (36, "B"),
    500: (279, "R1"),
    501: (222, "F"),
    502: (10, "B"),
    503: (178, "R2"),
    504: (501, "R0"),
    505: (471, "R0"),
    506: (394, "R0"),
    507: (270, "B"),
    508: (349, "P"),
    509: (149, "R1"),
    510: (45, "R1"),
    511: (23, "P"),
    512: (465, "B"),
    513: (507, "B"),
    514: (244, "R2"),
    515: (166, "B"),
    516: (140, "B"),
    517: (372, "B"),
    518: (152, "F"),
    519: (412, "B"),
    520: (261, "B"),
    521: (54, "R2"),
    522: (4, "F"),
    523: (157, "B"),
    524: (406, "R0"),
    525: (48, "R2"),
}


def _apply_transform(tensor, transform: str, device):
    """Apply a single consteval transformation to a host TTNN tensor."""
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM_MC)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)

    if transform == "B":
        return [t]

    if transform == "F":
        t2 = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        ttnn.deallocate(t, False)
        return [t2]

    if transform == "P":
        t2 = ttnn.permute(t, [1, 0], memory_config=DRAM_MC, pad_value=0.0)
        ttnn.deallocate(t, False)
        t3 = ttnn.typecast(t2, ttnn.DataType.FLOAT32, memory_config=DRAM_MC)
        ttnn.deallocate(t2, False)
        return [t3]

    if transform == "R0":
        t2 = ttnn.reshape(t, [1, 1, 1, 64, 2], memory_config=DRAM_MC)
        ttnn.deallocate(t, False)
        return [t2]

    if transform == "R1":
        t2 = ttnn.reshape(t, [1, 1, 3840], memory_config=DRAM_MC)
        ttnn.deallocate(t, False)
        return [t2]

    if transform == "R2":
        t2 = ttnn.reshape(t, [1, 3840], memory_config=DRAM_MC)
        ttnn.deallocate(t, False)
        return [t2]

    if transform == "R3":
        t2 = ttnn.reshape(t, [1, 2560], memory_config=DRAM_MC)
        ttnn.deallocate(t, False)
        return [t2]

    raise ValueError(f"Unknown transform type: {transform!r}")


def run_const_evals(inputs: list, device) -> dict:
    """Apply all consteval transformations and return the cache dict.

    Args:
        inputs: The 529-element inputs list from load_static_inputs().
        device: The (mesh) device to place tensors on.

    Returns:
        Dict keyed by "main_const_eval_{N}" — same format as the cached dict
        expected by graph._main().
    """
    cache = {}
    for ce_idx, (arg_idx, transform) in CONSTEVAL_MAP.items():
        result = _apply_transform(inputs[arg_idx], transform, device)
        cache[f"main_const_eval_{ce_idx}"] = result

    # ── Constant generators ───────────────────────────────────────────────────
    # These 9 entries are not loaded from model weights but computed from scratch.
    # They were originally inline constant functions in the generated graph.
    replicated = ttnn.ReplicateTensorToMesh(device)

    def _sf32(v):
        """Scalar [1,1] FLOAT32 TILE constant."""
        t = torch.tensor([[v]], dtype=torch.float32)
        return ttnn.from_torch(
            t,
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.Layout.TILE,
            device=device,
            memory_config=DRAM_MC,
            mesh_mapper=replicated,
        )

    def _sbf16(v):
        """Scalar [1,1] BFLOAT16 TILE constant."""
        t = torch.tensor([[v]], dtype=torch.bfloat16)
        return ttnn.from_torch(
            t,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=device,
            memory_config=DRAM_MC,
            mesh_mapper=replicated,
        )

    def _ids(vals):
        """1D position-ID list → [1, N] UINT32 ROW_MAJOR on-device tensor."""
        t = torch.tensor([vals], dtype=torch.int32)  # [1, N]
        return ttnn.from_torch(
            t,
            dtype=ttnn.DataType.UINT32,
            layout=ttnn.Layout.ROW_MAJOR,
            device=device,
            memory_config=DRAM_MC,
            mesh_mapper=replicated,
        )

    # ce_367: scalar 1000.0 BF16 — timestep scale factor (t_scale=1000 in PT)
    cache["main_const_eval_367"] = [_sbf16(1000.0)]

    # ce_219: sinusoidal frequencies [1, 128] F32 — TimestepEmbedder
    # freqs[i] = exp(-log(10000) * i / 128)
    half = 128
    freqs_ts = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half).unsqueeze(0)  # [1, 128]
    cache["main_const_eval_219"] = [
        ttnn.from_torch(
            freqs_ts,
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.Layout.TILE,
            device=device,
            memory_config=DRAM_MC,
            mesh_mapper=replicated,
        )
    ]

    # ce_208: eps=[1e-6, 1e-6, 1e-6] F32 — RMS norm epsilon for 3 norm sites
    #   [0] var_11: large-seq (IMG_PATCHES + CAP_TOKENS) RMS norm
    #   [1] var_12: caption token (2560-dim) RMS norm
    #   [2] var_13: QK norm (head_dim-128) RMS norm
    cache["main_const_eval_208"] = [_sf32(1e-6), _sf32(1e-6), _sf32(1e-6)]

    # ce_230: [1/3840, 1/3840] F32 — image token RMS norm scale (dim=3840)
    #   [0] var_15: keepdim=False path (shape [1024,1] after sum)
    #   [1] var_16: keepdim=True path (after all_gather)
    cache["main_const_eval_230"] = [_sf32(1.0 / 3840.0), _sf32(1.0 / 3840.0)]

    # ce_408: scalar 1/128 F32 — QK norm scale (head_dim=128)
    cache["main_const_eval_408"] = [_sf32(1.0 / 128.0)]

    # ce_454: scalar 1.0 BF16 — adaLN addend: scale_msa → (1 + scale_msa)
    cache["main_const_eval_454"] = [_sbf16(1.0)]

    # ce_88: scalar 1/2560 F32 — caption token RMS norm scale (dim=2560)
    cache["main_const_eval_88"] = [_sf32(1.0 / 2560.0)]

    # ── Position ID tensors ────────────────────────────────────────────────────
    from dit.model_ttnn import CAP_TOKENS as _CAP

    cap_len = _CAP

    # Caption: cap_len tokens.
    # pos_start=(1, 0, 0): caption F-positions are 1..cap_len; H/W positions are 0.
    cap_f_ids = list(range(1, cap_len + 1))
    cap_hw_ids = [0] * cap_len

    # Image: latent [16, 1, 64, 64] with patch_size=2, f_patch_size=1 →
    #   F_t=1, H_t=32, W_t=32 → 1024 tokens, no padding (1024 % 32 == 0).
    # pos_start=(cap_len+1, 0, 0).
    img_f_start = cap_len + 1
    img_f_ids = [img_f_start] * 1024
    img_h_ids = [h for h in range(32) for _ in range(32)]  # row 0..31, each ×32
    img_w_ids = list(range(32)) * 32  # col 0..31 cycling ×32

    # ce_96: caption F position IDs [1, cap_len] UINT32 — lookup into freqs_F
    cache["main_const_eval_96"] = [_ids(cap_f_ids)]

    # ce_158: image + caption position IDs — [F_ids, H_ids, W_ids, cap_HW_ids]
    #   [0] var_8[0]: image F IDs [1, 1024] — lookup into freqs_F
    #   [1] var_8[1]: image H IDs [1, 1024] — lookup into freqs_H
    #   [2] var_8[2]: image W IDs [1, 1024] — lookup into freqs_W
    #   [3] var_9:    caption H/W IDs [1, cap_len] — lookup into freqs_H and freqs_W
    cache["main_const_eval_158"] = [
        _ids(img_f_ids),
        _ids(img_h_ids),
        _ids(img_w_ids),
        _ids(cap_hw_ids),
    ]

    return cache
