# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
# Perf experiment copy of all_gather_rms_norm (far_slice_fabric_mcast). AGRMS_MCAST=1 (default): multicast transport.
from .all_gather_rms_norm import all_gather_rms_norm

__all__ = ["all_gather_rms_norm"]
