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

GROUP_ROWS = 5
MAX_GROUPS = GRID_ROWS // GROUP_ROWS

NORM_CHUNK_TILES = 16
NORM_DEST_TILES = 4
NORM_IN_CHUNKS = 4
NORM_OUT_CHUNKS = 3
NORM_READS_IN_FLIGHT = 3
N_DEPTH = 2
N_ROW_TILES = HIDDEN_TILES + 1

A_COLUMNS = 7
A_ACTIVE = A_COLUMNS * GROUP_ROWS - 1
A_TILES_PER_CORE = 4
A_OUT_DEPTH = 8
GATHER_TILES = A_COLUMNS * A_TILES_PER_CORE
GATHER_DEPTH = 3

B_FIRST_X = A_COLUMNS
B_COLUMNS = 4
B_ROWS = GROUP_ROWS
B_CORES = B_COLUMNS * B_ROWS
B_TILES_PER_CORE = HIDDEN_TILES // B_CORES
A_DEPTH = 3

WEIGHT_CHUNK_ROWS = 8
WEIGHT_READS_IN_FLIGHT = 3

PACKET_TILES = 4
PACKET_BYTES = PACKET_TILES * BF16_TILE_BYTES
PACKETS_PER_SEGMENT = B_TILES_PER_CORE // PACKET_TILES
LANDING_DEPTH = 3
OWNER_LAG = 2
MESSAGE_BYTES = 32
RELAY_SLOT_BYTES = MESSAGE_BYTES + PACKET_BYTES
RELAY_SLOTS = 1

# B column (x - B_FIRST_X) -> block: the output block it computes belongs to chip (chip + block) % CHIPS
BLOCK_OF_COLUMN = [2, 1, 3, 0]
COLUMN_OF_BLOCK = [BLOCK_OF_COLUMN.index(block) for block in range(B_COLUMNS)]
OWNER_BLOCK = 0

# block -> (ring direction, hops, relay index) used to send a partial segment to its owner chip
PARTIAL_ROUTE = {1: (0, 1, 1), 2: (0, 2, 0), 3: (1, 1, 2)}
RELAY_COUNT = 4
RELAY_CORE_LIST = [(B_FIRST_X, 4), (B_FIRST_X + 1, 4), (B_FIRST_X + 2, 4), (B_FIRST_X + 2, 3)]
FLAG_BYTES = 16
FLAG_WORDS = 16

# CB_N on the A rectangle (norm core included) and CB_A on the B cores are each the first CB of their cores, so they share an L1 address;
# the landing buffers start at the second CB of every B core and the relay slots (other B cores) follow the first landing buffer, so every B core can address both
CB_N = 0
CB_A = 0
CB_NORM_WEIGHT = 1
CB_NORM_IN = 2
CB_NORM_OUT = 3
CB_NORM_SCALER = 4
CB_NORM_ONES = 5
CB_NORM_SUM = 6
CB_NORM_RSTD = 7
CB_NORM_RSTD_OUT = 8
CB_NORM_STARTUP = 9
CB_WG = 10
CB_A_OUT = 11
CB_WD = 12
CB_PARTIAL_OWN = 13
CB_H = 14
CB_Y = 15
CB_SEND = 16
CB_LANDING = 17
CB_RELAY_SLOTS = 21
CB_MESSAGE = 22
CB_GATHER = 23

SEM_N_FREE = 0
SEM_N_VALID = 0
SEM_A_FREE = N_DEPTH
SEM_GATHER_VALID = SEM_A_FREE + A_DEPTH
SEM_GATHER_FREE = SEM_GATHER_VALID + GATHER_DEPTH
SEM_A_WEIGHTS = SEM_GATHER_FREE + GATHER_DEPTH
SEM_A_VALID = 0
SEM_LAND_READY = SEM_A_VALID + A_DEPTH
SEM_CREDIT_PARTIAL = SEM_LAND_READY + CHIPS - 1
SEM_CREDIT_Y = SEM_CREDIT_PARTIAL + 1
SEM_RELAY_FREE = SEM_CREDIT_Y + CHIPS - 1
SEM_B_WEIGHTS = SEM_RELAY_FREE + 2


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


def core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def first_row(group):
    return group * GROUP_ROWS


def a_rect(group):
    return Block(0, first_row(group), A_COLUMNS - 1, first_row(group) + GROUP_ROWS - 1)


def b_rect(group):
    return Block(B_FIRST_X, first_row(group), B_FIRST_X + B_COLUMNS - 1, first_row(group) + GROUP_ROWS - 1)


def norm_core(group):
    return (0, first_row(group))


def a_cores(group):
    return [
        (x, first_row(group) + row)
        for row in range(GROUP_ROWS)
        for x in range(A_COLUMNS)
        if (x, first_row(group) + row) != norm_core(group)
    ]


def a_row_cores(group, row):
    return [core for core in a_cores(group) if core[1] == first_row(group) + row]


def a_collectors(group):
    return Block(A_COLUMNS - 1, first_row(group), A_COLUMNS - 1, first_row(group) + GROUP_ROWS - 1)


def b_core(group, block, row):
    return B_FIRST_X + COLUMN_OF_BLOCK[block], first_row(group) + row


def peer_index(peer, chip):
    return ((peer - chip) % CHIPS) - 1


assert A_ACTIVE * A_TILES_PER_CORE == INTERMEDIATE_TILES
assert B_TILES_PER_CORE * B_CORES == HIDDEN_TILES
assert HIDDEN_TILES % NORM_CHUNK_TILES == 0
assert all(x != B_FIRST_X + COLUMN_OF_BLOCK[OWNER_BLOCK] for x, _ in RELAY_CORE_LIST)
