# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device query API. Experimental; may change.

``get_info(mesh_device, info.l1_alignment)`` returns a device property for the whole mesh (every local device must
agree), ``get_info(mesh_device, info.l1_alignment, coord)`` for one device, and
``get_info_per_device(mesh_device, info.l1_alignment)`` a ``{MeshCoordinate: value}`` dict for every local device.

The tags (``l1_alignment``, ``dram_alignment``, ``architecture``, ``architecture_name``) are generated from the C++
``experimental::info::all_tags`` list, so a property added there appears here with no change to this file.
"""

from ttnn._ttnn.multi_device.experimental import get_info, get_info_per_device
from ttnn._ttnn.multi_device.experimental import info as _native_info

InfoTag = _native_info.InfoTag
all_tags = _native_info.all_tags

globals().update({tag.name: tag for tag in all_tags()})

__all__ = ["InfoTag", "all_tags", "get_info", "get_info_per_device", *(tag.name for tag in all_tags())]
