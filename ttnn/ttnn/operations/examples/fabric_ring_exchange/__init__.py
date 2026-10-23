# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from .program_descriptor import (
    VARIANTS,
    Resources,
    exchange_plan,
    expected_landing,
    idle_cores,
    port_cores,
    ring_coords,
    ring_exchange,
)

__all__ = [
    "VARIANTS",
    "Resources",
    "exchange_plan",
    "expected_landing",
    "idle_cores",
    "port_cores",
    "ring_coords",
    "ring_exchange",
]
