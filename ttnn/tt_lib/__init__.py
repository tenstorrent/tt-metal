# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
from pathlib import Path
from loguru import logger


# def _has_not_found(target_so):
#     if not os.path.exists(target_so):
#         logger.trace(f"Shared library {target_so} does not exists")
#         return False
#     cmd = f"ldd {target_so}"
#     result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
#     return "not found" in result.stdout


# def _check_so_rpath(so_name, new_rpath):
#     directory = Path(__file__).parent
#     check_f = directory / f".rpath_checked_{so_name}"
#     if os.path.exists(check_f):
#         return
#     target_so = None
#     for f in os.listdir(directory):
#         if f.startswith(so_name) and f.endswith(".so"):
#             target_so = directory / f
#             break
#     if not target_so:
#         logger.trace(f"Cannot find shared library which name starts with {so_name}")
#         return

#     if _has_not_found(target_so):
#         subprocess.check_call(f"patchelf --set-rpath {new_rpath} {target_so}", shell=True)
#     subprocess.check_call(f"touch {check_f}", shell=True)


# def _check_so_rpath_in_build_lib():
#     directory = Path(__file__).parent / "build/lib"
#     check_f = directory / ".rpath_checked"
#     if not os.path.exists(directory):
#         logger.trace(f"Directory {directory} does not exists")
#         return
#     if os.path.exists(check_f):
#         return
#     import subprocess

#     ttnn_so = directory / "_ttnn.so"
#     metal_so = directory / "libtt_metal.so"

#     new_rpath = directory
#     if _has_not_found(ttnn_so):
#         subprocess.check_call(f"patchelf --set-rpath {new_rpath} {ttnn_so}", shell=True)
#     if _has_not_found(metal_so):
#         subprocess.check_call(f"patchelf --set-rpath {new_rpath} {metal_so}", shell=True)
#     subprocess.check_call(f"touch {check_f}", shell=True)


# site_pkgs_tt_lib = Path(__file__).parent

# if "ARCH_NAME" not in os.environ or os.environ["ARCH_NAME"] == "":
#     arch_name_file = site_pkgs_tt_lib / ".ARCH_NAME"
#     if os.path.isfile(arch_name_file):
#         with open(arch_name_file) as f:
#             os.environ["ARCH_NAME"] = f.readline().strip()
# if "TT_METAL_HOME" not in os.environ or os.environ["TT_METAL_HOME"] == "":
#     # Workaround: treat $SITE_PACKAGES as TT_METAL_HOME
#     os.environ["TT_METAL_HOME"] = str(site_pkgs_tt_lib.parent)
# # jit build needs linker script under $TT_METAL_HOME/build/hw/toolchain/,
# # so when TT_METAL_HOME is site-packags,
# # it needs to softlink build/ from site-packages/tt_lib
# build_soft_link = site_pkgs_tt_lib / ".." / "build"
# build_soft_link_src = site_pkgs_tt_lib / "build"
# if not os.path.exists(build_soft_link) and os.path.exists(build_soft_link_src):
#     os.symlink(build_soft_link_src, build_soft_link)

# _check_so_rpath_in_build_lib()
# _check_so_rpath("_C", site_pkgs_tt_lib / "build" / "lib")

import sys

from ttnn._ttnn.deprecated import tensor, device, profiler, operations

import tt_lib.fallback_ops
import tt_lib.fused_ops
