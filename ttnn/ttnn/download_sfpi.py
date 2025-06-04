# SPDX-FileCopyrightText: © 2025  Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import json
import hashlib
import urllib.request
import tarfile
from loguru import logger


def download_sfpi(sfpi_json_path, sfpi_target_dir):
    # Load release info
    try:
        with open(sfpi_json_path, "r") as f:
            sfpi_releases = json.load(f)
    except FileNotFoundError:
        logger.error(f"SFPI version JSON not found: {sfpi_json_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in: {sfpi_json_path}")
        sys.exit(1)

    # Detect host key
    arch = os.uname().machine
    os_name = os.uname().sysname
    key = f"{arch}_{os_name}"

    if key not in sfpi_releases:
        logger.error(f"SFPI binaries for {key} not available")
        sys.exit(1)

    sfpi_file, expected_md5 = sfpi_releases[key]
    version_tag, filename = sfpi_file.split("/")
    url = f"https://github.com/tenstorrent/sfpi/releases/download/{version_tag}/{filename}"

    # Compute paths
    download_path = os.path.join(sfpi_target_dir, filename)

    os.makedirs(sfpi_target_dir, exist_ok=True)

    logger.info(f"Downloading {url} to {download_path}")
    urllib.request.urlretrieve(url, download_path)

    # Verify MD5
    with open(download_path, "rb") as f:
        file_data = f.read()
        actual_md5 = hashlib.md5(file_data).hexdigest()

    if actual_md5 != expected_md5:
        logger.error(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")
        sys.exit(1)

    logger.info(f"Extracting {download_path} to {sfpi_target_dir}")
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path=sfpi_target_dir)

    logger.info(f"SFPI downloaded and extracted to {sfpi_target_dir}")


def main():
    # Default path (update as needed)
    sfpi_json_path = os.path.join(os.path.dirname(__file__), "build", "sfpi-version.json")
    sfpi_target_dir = os.path.join(os.path.dirname(__file__), "runtime")
    download_sfpi(sfpi_json_path, sfpi_target_dir)


if __name__ == "__main__":
    main()
