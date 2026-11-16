# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Port-assignment policies for the W stream: per core, (slot_bits, rotate).

slot s of bank b = the bank's s-th translated DRAM port (translated y = y0 + s). Returns (bits_a, rotate, bits_b):
rotate 1 = read r uses slot a; 2 = a / b alternate by r; 3 = (a + r) % 3.

PHYS_ROWS / PREF_SLOT: this p100a (DRAM channel 2 harvested, pre-relocation CMFW), measured by test_probe_ports.py:
bank -> NoC0 physical row of slots 0..2 (soc descriptor `dram:` list of the bank's channel) and the slot of the
preferred NoC0 endpoint. West banks 0-2 sit at x=0, east banks 3-6 at x=9.
"""

PHYS_ROWS = {
    0: (0, 1, 11),  # ch0
    1: (2, 10, 3),  # ch1
    2: (5, 7, 6),  # ch3
    3: (0, 1, 11),  # ch4
    4: (2, 10, 3),  # ch5
    5: (9, 4, 8),  # ch6
    6: (5, 7, 6),  # ch7
}
PHYS_X = {0: 0, 1: 0, 2: 0, 3: 9, 4: 9, 5: 9, 6: 9}
PREF_SLOT = {0: 2, 1: 0, 2: 0, 3: 2, 4: 2, 5: 2, 6: 2}
NOC1_SLOT = {b: 1 for b in range(7)}
FREE_SLOT = {b: 3 - PREF_SLOT[b] - NOC1_SLOT[b] for b in range(7)}
NOC_X, NOC_Y = 17, 12


def _bits(slots):
    return sum((s % 3) << (2 * b) for b, s in slots.items())


def _hops(b, s, cx, cy):
    """NoC0 hops of a response from port (b, s) to core (cx, cy): east first, then south (both wrap)."""
    return (cx - PHYS_X[b]) % NOC_X + (cy - PHYS_ROWS[b][s]) % NOC_Y


def _policy(name, idx, logical, virt):
    banks = range(7)
    if name in ("acc", "pref"):
        return _bits(PREF_SLOT), 1, 0
    if name == "free":
        return _bits(FREE_SLOT), 1, 0
    if name == "noc1ep":
        return _bits(NOC1_SLOT), 1, 0
    if name == "fix11":
        return _bits({**PREF_SLOT, 0: 0, 3: 1}), 1, 0
    if name == "rot":
        return _bits(PREF_SLOT), 3, 0
    if name == "rot2":
        return _bits(PREF_SLOT), 2, _bits(FREE_SLOT)
    if name == "rot2_core":
        a, b = (PREF_SLOT, FREE_SLOT) if idx % 2 == 0 else (FREE_SLOT, PREF_SLOT)
        return _bits(a), 2, _bits(b)
    if name == "core2":
        return _bits(PREF_SLOT if idx % 2 == 0 else FREE_SLOT), 1, 0
    if name == "rot_core":
        return _bits({b: PREF_SLOT[b] + idx for b in banks}), 3, 0
    if name == "core3":
        return _bits({b: PREF_SLOT[b] + idx for b in banks}), 1, 0
    if name == "row3":
        return _bits({b: PREF_SLOT[b] + logical.y for b in banks}), 1, 0
    if name == "nearest":
        return _bits({b: min(range(3), key=lambda s: _hops(b, s, virt.x, virt.y)) for b in banks}), 1, 0
    if name == "farthest":
        return _bits({b: max(range(3), key=lambda s: _hops(b, s, virt.x, virt.y)) for b in banks}), 1, 0
    raise ValueError(name)


def policy(name, idx, logical, virt):
    """[bits_a, rotate, bits_b] for core number idx (assignment order) at logical / virtual coords."""
    return list(_policy(name, idx, logical, virt))
