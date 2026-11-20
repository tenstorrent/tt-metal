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

CHIPS = 4
GRID_COLUMNS = 11
GRID_ROWS = 10

NORM_CORES = 2
NORM_DEST_TILES = 4
MCAST_CHUNK_TILES = 16
N_DEPTH = 4
N_ROW_TILES = HIDDEN_TILES + 1

A_COLUMNS = 7
A_ACTIVE = 68
A_COLLECTORS = GRID_ROWS
A_TILES_PER_CORE = 2
A_OUT_DEPTH = 4
GATHER_TILES = A_COLUMNS * A_TILES_PER_CORE

B_FIRST_X = 7
B_COLUMNS = 4
B_ROWS = GRID_ROWS
B_CORES = B_COLUMNS * B_ROWS
B_TILES_PER_CORE = HIDDEN_TILES // B_CORES
A_DEPTH = 4

SEGMENT_BYTES = B_TILES_PER_CORE * BF16_TILE_BYTES
LANDING_DEPTH = 6
OWNER_LAG = 3
MESSAGE_BYTES = 32
RELAY_SLOT_BYTES = MESSAGE_BYTES + SEGMENT_BYTES
RELAY_SOURCE_SLOTS = 15

# B column (x - B_FIRST_X) -> group: the output block it computes belongs to chip (chip + group) % CHIPS
GROUP_OF_COLUMN = [2, 1, 3, 0]
COLUMN_OF_GROUP = [GROUP_OF_COLUMN.index(group) for group in range(B_COLUMNS)]
OWNER_GROUP = 0

# group -> (ring direction, hops, relay index) used to send a partial segment to its owner chip
PARTIAL_ROUTE = {1: (0, 1, 1), 2: (0, 2, 0), 3: (1, 1, 2)}
RELAY_COUNT = 4
RELAY_ROW = GRID_ROWS - 1
FLAG_BYTES = 16
FLAG_WORDS = 16

# CB_A on the B cores, CB_NORM_OUT on the norm cores and CB_N on the A cores are each the first CB of their cores, so they share an L1 address
CB_NORM_WEIGHT = 0
CB_NORM_IN = 1
CB_N = 2
CB_NORM_OUT = 8
CB_NORM_SCALER = 3
CB_NORM_ONES = 4
CB_NORM_SUM = 5
CB_NORM_RSTD = 6
CB_NORM_STARTUP = 16
CB_WG = 9
CB_A_OUT = 10
CB_A = 11
CB_WD = 12
CB_PARTIAL_OWN = 13
CB_H = 14
CB_Y = 15
CB_SEND = 17
CB_LANDING = 18
CB_RELAY_SLOTS = 22
CB_MESSAGE = 23
CB_GATHER = 24

SEM_N_FREE = 0
SEM_N_VALID = 0
SEM_A_FREE = N_DEPTH
SEM_GATHER_VALID = SEM_A_FREE + A_DEPTH
SEM_GATHER_FREE = SEM_GATHER_VALID + A_DEPTH
SEM_A_VALID = 0
SEM_LAND_READY = SEM_A_VALID + A_DEPTH
SEM_CREDIT_PARTIAL = SEM_LAND_READY + CHIPS - 1
SEM_CREDIT_Y = SEM_CREDIT_PARTIAL + 1
SEM_RELAY_FREE = SEM_CREDIT_Y + CHIPS - 1


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


GRID = Block(0, 0, GRID_COLUMNS - 1, GRID_ROWS - 1)
B_GRID = Block(B_FIRST_X, 0, B_FIRST_X + B_COLUMNS - 1, GRID_ROWS - 1)
NORM_CORE_LIST = [(0, 0), (1, 0)]
A_CORE_LIST = [(x, y) for y in range(GRID_ROWS) for x in range(A_COLUMNS) if (x, y) not in NORM_CORE_LIST]
RELAY_CORE_LIST = [(B_FIRST_X + column, RELAY_ROW) for column in range(RELAY_COUNT)]

assert len(A_CORE_LIST) == A_ACTIVE
assert A_ACTIVE * A_TILES_PER_CORE == INTERMEDIATE_TILES
assert B_TILES_PER_CORE * B_CORES == HIDDEN_TILES
assert NORM_CORES == len(NORM_CORE_LIST)


def core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def a_row_cores(row):
    return [core for core in A_CORE_LIST if core[1] == row]


def b_core(group, row):
    return B_FIRST_X + COLUMN_OF_GROUP[group], row


def peer_index(peer, chip):
    return ((peer - chip) % CHIPS) - 1
