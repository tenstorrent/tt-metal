# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""python -m ttnn.matmul_model: downloads the matmul config model blob into ttnn's cache and prints its path."""

from ttnn.matmul_model import download

print(download())
