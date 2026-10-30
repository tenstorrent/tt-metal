# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from ttnn.operations.layer_norm_tile import default_compute_kernel_config, layer_norm

__all__ = ["layer_norm", "default_compute_kernel_config"]
