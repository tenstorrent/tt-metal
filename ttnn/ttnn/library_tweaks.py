# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
from loguru import logger


def _setup_env(site_pkgs_ttnn):
    if "ARCH_NAME" not in os.environ or os.environ["ARCH_NAME"] == "":
        arch_name_file = site_pkgs_ttnn / ".ARCH_NAME"
        if os.path.isfile(arch_name_file):
            with open(arch_name_file) as f:
                os.environ["ARCH_NAME"] = f.readline().strip()
    if "TT_METAL_HOME" not in os.environ or os.environ["TT_METAL_HOME"] == "":
        # Workaround: treat $SITE_PACKAGES as TT_METAL_HOME
        os.environ["TT_METAL_HOME"] = str(site_pkgs_ttnn.parent)

    # jit build needs linker script under $TT_METAL_HOME/hw/toolchain/,
    # so when TT_METAL_HOME is site-packages,
    # it needs to softlink build/ from site-packages/tt_lib


def setup_ttnn_so():
    site_pkgs_ttnn = Path(__file__).parent

    _setup_env(site_pkgs_ttnn)
