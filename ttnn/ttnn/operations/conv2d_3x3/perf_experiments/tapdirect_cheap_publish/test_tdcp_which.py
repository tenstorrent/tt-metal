# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which op copy the retargeted tests load (TDCP_COPY_DIR, default op_copy). Host-only."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tdcp_copy import copy_modules  # noqa: E402


def test_tdcp_which():
    op_mod, pd_mod = copy_modules()
    want = os.environ.get("TDCP_COPY_DIR", "op_copy")
    assert Path(pd_mod.__file__).parent.name == want, pd_mod.__file__
    assert Path(pd_mod.KERNEL_DIR).parent.name == want, pd_mod.KERNEL_DIR
    assert pd_mod.PADDED_SLAB is True
    assert hasattr(pd_mod, "LAST_PADDED_INFO") == (want == "op_copy")
