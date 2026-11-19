# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import ttnn

TILE = 32
HIDDEN_TILES = 160
INTERMEDIATE_TILES = 136
GATE_UP_TILES = 2 * INTERMEDIATE_TILES

BF16_TILE_BYTES = 2048
BFP4_TILE_BYTES = 576
ACT_TILE_BYTES = 1088

NORM_CORES = 4
NORM_CHUNK_TILES = 8
NORM_DEST_TILES = 4
MCAST_CHUNK_TILES = 16

A_COLUMNS = 7
A_ROWS = 10
A_ACTIVE = 68
A_TILES_PER_CORE = 2
B_FIRST_X = 7
B_COLUMNS = 4
B_FIRST_Y = 2
B_ROWS = 8
B_CORES = B_COLUMNS * B_ROWS
B_TILES_PER_CORE = HIDDEN_TILES // B_CORES
K_GROUPS = B_COLUMNS
K_TILES_PER_B = INTERMEDIATE_TILES // K_GROUPS
N_TILES_PER_B = HIDDEN_TILES // B_ROWS
A_PER_K_GROUP = A_ACTIVE // K_GROUPS
KRED_DEPTH = 3
FLAG_BYTES = 16
NORM_Y = 1
RELAY_Y = 0

assert A_ACTIVE * A_TILES_PER_CORE == INTERMEDIATE_TILES
assert B_TILES_PER_CORE * K_GROUPS == N_TILES_PER_B
assert A_PER_K_GROUP * A_TILES_PER_CORE == K_TILES_PER_B


@dataclass(frozen=True)
class Block:
    x_start: int
    y_start: int
    x_end: int
    y_end: int

    @property
    def cores(self):
        return (self.x_end - self.x_start + 1) * (self.y_end - self.y_start + 1)

    def core_range_set(self):
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(self.x_start, self.y_start), ttnn.CoreCoord(self.x_end, self.y_end))]
        )

    def logical(self, index):
        width = self.x_end - self.x_start + 1
        return self.x_start + index % width, self.y_start + index // width


A_GRID = Block(0, 0, A_COLUMNS - 1, A_ROWS - 1)


def a_core(index):
    return index // A_ROWS, index % A_ROWS


def a_active_cores():
    full_columns = A_ACTIVE // A_ROWS
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_columns - 1, A_ROWS - 1)),
            ttnn.CoreRange(ttnn.CoreCoord(full_columns, 0), ttnn.CoreCoord(full_columns, A_ACTIVE % A_ROWS - 1)),
        ]
    )


B_GRID = Block(B_FIRST_X, B_FIRST_Y, B_FIRST_X + B_COLUMNS - 1, B_FIRST_Y + B_ROWS - 1)
NORM_GRID = Block(B_FIRST_X, NORM_Y, B_FIRST_X + NORM_CORES - 1, NORM_Y)
RELAY_COUNT = 4
RELAY_GRID = Block(B_FIRST_X, RELAY_Y, B_FIRST_X + RELAY_COUNT - 1, RELAY_Y)

CB_NORM_WEIGHT = 0
CB_NORM_IN = 1
CB_NORM_OUT = 2
CB_NORM_SCALER = 3
CB_NORM_SQUARES = 4
CB_NORM_SUM = 5
CB_NORM_RSTD = 6
CB_NORM_SCALED = 7
CB_NORM_STARTUP = 16
CB_N = 8
CB_WG = 9
CB_A_OUT = 10
CB_A = 11
CB_WD = 12
CB_PARTIAL_OWN = 13
CB_H = 14
CB_Y = 15
CB_SEND = 17
CB_RS = 18
CB_AG = 21
CB_RELAY_SLOTS = 22
CB_MESSAGE = 23
CB_KOWN = 24
CB_KOUT = 25
CB_KRED = 26
CB_FLAGS = 29

CHIPS = 4
RING_DEPTH = 3
OWNER_LAG = 2
REDUCE_LAG = 0
RELAY_SOURCES = 16
SEGMENT_BYTES = B_TILES_PER_CORE * BF16_TILE_BYTES
MESSAGE_BYTES = 32
RELAY_SLOT_BYTES = MESSAGE_BYTES + SEGMENT_BYTES

N_DEPTH = 4
SEM_N_FREE = 0
SEM_N_VALID = 0
SEM_A_FREE = N_DEPTH
SEM_A_VALID = 0
SEM_RS_READY = 2
SEM_AG_READY = 5
SEM_CREDIT_RS = 8
SEM_CREDIT_AG = 11
SEM_RELAY_FREE = 14
