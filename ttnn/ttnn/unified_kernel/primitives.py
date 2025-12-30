# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Primitive definitions for unified kernels.
"""

from enum import IntEnum
from typing import Optional
import ttnn


class Role(IntEnum):
    """Core role in unified kernel execution."""

    READER = 0
    COMPUTE = 1
    WRITER = 2
    MCAST_SENDER = 3
    MCAST_RECEIVER = 4


class BufferMode(IntEnum):
    """Buffer mode for circular buffers."""

    SINGLE = 1  # Single buffer (saves L1 memory)
    DOUBLE = 2  # Double buffer (better performance, default)


class McastGroup:
    """Represents a multicast group with sender and receivers."""

    def __init__(
        self,
        name: str,
        sender: ttnn.CoreCoord,
        receivers: ttnn.CoreRangeSet,
        noc: Optional[ttnn.NOC] = None,
    ):
        self.name = name
        self.sender = sender
        self.receivers = receivers
        self.noc = noc or ttnn.NOC.NOC_1

        # Calculate if sender is part of receiver grid
        self.is_part_of_receiver_grid = receivers.contains(sender)

        # Calculate number of destinations
        receiver_cores = ttnn.corerange_to_cores(receivers)
        self.num_dests = len(receiver_cores)

        # If sender is in receiver grid, exclude it from count
        if self.is_part_of_receiver_grid:
            self.num_dests -= 1
