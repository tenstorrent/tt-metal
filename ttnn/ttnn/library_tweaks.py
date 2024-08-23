# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from loguru import logger
from importlib.metadata import version


def write_metal_version_to_file(version_file, metal_version):
    assert (
        metal_version
    ), f"Version of ttnn / tt-metal seems to be empty despite being installed from wheel. If you did not install from a wheel, please file an issue immediately"

    with open(version_file, "w") as f:
        f.write(metal_version)


def get_metal_version_from_file(version_file):
    assert version_file.is_file(), f"{version_file} seems to not exist"

    with open(version_file, "r") as f:
        metal_version = f.read().strip()

    return metal_version


def prepare_dir_as_metal_home(ttnn_package_path, metal_home):
    metal_home.mkdir(exist_ok=True)

    version_file = metal_home / ".METAL_VERSION"
    current_version = version("metal-libs").strip()

    runtime_src = ttnn_package_path.parent / "runtime"
    assert (
        runtime_src.is_dir()
    ), f"{runtime_src} seems to not exist as a directory. This should have been packaged during wheel creation"
    runtime_dest = metal_home / "runtime"

    tt_metal_src = ttnn_package_path.parent / "tt_metal"
    tt_metal_dest = metal_home / "tt_metal"

    ttnn_src = ttnn_package_path
    ttnn_dest = metal_home / "ttnn"

    if version_file.exists():
        last_used_version = get_metal_version_from_file(version_file)

        if last_used_version == current_version:
            assert (
                runtime_dest.is_dir()
            ), f"A .METAL_VERSION file exists in current working directory, but no {runtime_dest} directory within it. Your installation may be corrupted."
            logger.debug(f"Existing installation of {current_version} detected")
            return
        else:
            # TODO: Assert newer version
            logger.debug(
                f"Existing installation for {last_used_version} detected, but overriding runtime assets for version {current_version}. Consider deleting all generated folders"
            )

    logger.debug(f"Preparing {metal_home} as TT_METAL_HOME in a production environment")

    write_metal_version_to_file(version_file, current_version)

    # TODO: Assert these are ok, and that if the symlinks already exist then don't do anything
    runtime_dest.symlink_to(runtime_src)
    tt_metal_dest.symlink_to(tt_metal_src)
    ttnn_dest.symlink_to(ttnn_src)


def _setup_env(ttnn_package_path, cwd):
    if "ARCH_NAME" not in os.environ or os.environ["ARCH_NAME"] == "":
        arch_name_file = ttnn_package_path / ".ARCH_NAME"

        assert (
            arch_name_file.is_file()
        ), f".ARCH_NAME is not a file, so architecture cannot be determined. Are you installing ttnn from source? If you are installing and running from source, please set the ARCH_NAME environment variable."

        with open(arch_name_file) as f:
            os.environ["ARCH_NAME"] = f.readline().strip()

    if "TT_METAL_HOME" not in os.environ or os.environ["TT_METAL_HOME"] == "":
        # Workaround: treat cwd / ttnn_links as TT_METAL_HOME and copy assets to it
        metal_home = cwd / ".ttnn_runtime_artifacts"
        prepare_dir_as_metal_home(ttnn_package_path, metal_home)
        os.environ["TT_METAL_HOME"] = str(metal_home)

    # jit build needs linker script under $TT_METAL_HOME/hw/toolchain/,
    # so when TT_METAL_HOME is site-packages,
    # it needs to softlink build/ from site-packages/tt_lib


def setup_ttnn_so():
    ttnn_package_path = Path(__file__).resolve().parent

    cwd = Path(os.getcwd())

    _setup_env(ttnn_package_path, cwd)
