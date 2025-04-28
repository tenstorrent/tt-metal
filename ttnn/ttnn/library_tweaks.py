# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from loguru import logger
from importlib.metadata import version
import sys

from .download_sfpi import download_sfpi

# Configuration flags
# Allows us to chage our minds later
SFPI_IS_BUNDLED = False  # If true, the SFPI compiler is bundled with the wheel
SFPI_IS_SYSTEM_PACKAGE = not SFPI_IS_BUNDLED  # If true, the SFPI compiler is installed via a system package manager
AUTO_DOWNLOAD_SFPI = (
    not SFPI_IS_BUNDLED and not SFPI_IS_SYSTEM_PACKAGE
)  # If true, the SFPI compiler is downloaded if it is not found


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
    current_version = version("ttnn").strip()

    # Firmware binaries need to be present
    runtime_hw_src = ttnn_package_path / "runtime" / "hw"
    if not runtime_hw_src.is_dir():
        logger.error(
            f"{runtime_hw_src} seems to not exist as a directory. This should have been packaged during wheel creation"
        )
        sys.exit(1)
    runtime_hw_dest = metal_home / "runtime" / "hw"

    # SFPI compiler needs to be present if we are bundling it
    runtime_sfpi_src = ttnn_package_path / "runtime" / "sfpi"
    runtime_sfpi_dest = metal_home / "runtime" / "sfpi"
    if SFPI_IS_BUNDLED:
        if not runtime_sfpi_src.is_dir():
            logger.error(
                f"{runtime_sfpi_src} seems to not exist as a directory. This should have been packaged during wheel creation"
            )
            sys.exit(1)

    # If we are using a system package, we need to check that the SFPI compiler is installed
    if SFPI_IS_SYSTEM_PACKAGE:
        system_sfpi_path = Path("/opt/tenstorrent/sfpi")
        if not system_sfpi_path.is_dir():
            logger.error(
                f"SFPI system package not found at {system_sfpi_path}. Please install the SFPI system package."
            )
            sys.exit(1)

    tt_metal_src = ttnn_package_path / "tt_metal"
    tt_metal_dest = metal_home / "tt_metal"

    ttnn_src = ttnn_package_path
    ttnn_dest = metal_home / "ttnn"

    if version_file.exists():
        last_used_version = get_metal_version_from_file(version_file)

        if last_used_version == current_version:
            if not runtime_hw_dest.is_dir():
                logger.error(
                    f"A .METAL_VERSION file exists in current working directory, but no {runtime_hw_dest} directory within it. Your installation may be corrupted."
                )
                sys.exit(1)
            logger.debug(f"Existing installation of {current_version} detected")
            return
        else:
            # TODO: Assert newer version
            logger.debug(
                f"Existing installation for {last_used_version} detected, but overriding runtime assets for version {current_version}. Consider deleting all generated folders"
            )

    logger.debug(f"Preparing {metal_home} as TT_METAL_HOME in a production environment")

    write_metal_version_to_file(version_file, current_version)

    if not runtime_hw_dest.exists():
        runtime_hw_dest.parent.mkdir(parents=True, exist_ok=True)  # mkdir runtime
        runtime_hw_dest.symlink_to(runtime_hw_src)

    if SFPI_IS_BUNDLED and not runtime_sfpi_dest.exists():
        runtime_sfpi_dest.parent.mkdir(parents=True, exist_ok=True)  # mkdir runtime
        runtime_sfpi_dest.symlink_to(runtime_sfpi_src)

    if not tt_metal_dest.exists():
        tt_metal_dest.symlink_to(tt_metal_src)

    if not ttnn_dest.exists():
        ttnn_dest.symlink_to(ttnn_src)

    if AUTO_DOWNLOAD_SFPI:
        sfpi_json_path = ttnn_package_path / "build" / "lib" / "sfpi-version.json"
        runtime_sfpi_dest = metal_home / "runtime"
        download_sfpi(sfpi_json_path, runtime_sfpi_dest)


def _is_non_existent_or_empty_env_var(env_var_name):
    assert isinstance(env_var_name, str)
    return env_var_name not in os.environ or not os.environ[env_var_name]


def _setup_env(ttnn_package_path, cwd):
    if _is_non_existent_or_empty_env_var("TT_METAL_HOME"):
        # Workaround: treat cwd / ttnn_links as TT_METAL_HOME and copy/symlink assets to it
        metal_home = cwd / ".ttnn_runtime_artifacts"
        prepare_dir_as_metal_home(ttnn_package_path, metal_home)
        os.environ["TT_METAL_HOME"] = str(metal_home)


def setup_ttnn_so():
    ttnn_package_path = Path(__file__).resolve().parent

    cwd = Path(os.getcwd())

    _setup_env(ttnn_package_path, cwd)
