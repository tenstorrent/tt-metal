# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lego: fused programs composed on the host from kernel blocks ported from existing ops."""

from .compose import Block, LegoProgram, core_range_set, cores_of, mesh_descriptor  # noqa: F401
