# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import glob
import shutil
from typing import List, Tuple


def get_cache_dirs() -> List[str]:
    """Get list of cache directories used by SDXL server"""
    # Use default tt-metal cache (shared with tt-media-server) unless TT_METAL_CACHE is set
    tt_metal_cache = os.environ.get("TT_METAL_CACHE", os.path.expanduser("~/.cache/tt-metal-cache"))
    return [
        tt_metal_cache,
        os.path.expanduser("~/.cache/ttnn/models"),
    ]


def validate_cache() -> Tuple[bool, List[str]]:
    """
    Validate cache integrity by checking for corruption indicators

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    for cache_dir in get_cache_dirs():
        if not os.path.exists(cache_dir):
            continue

        # Check for incomplete files (tmp files)
        tmp_files = glob.glob(f"{cache_dir}/**/*.tmp", recursive=True)
        if tmp_files:
            issues.append(f"Found {len(tmp_files)} incomplete .tmp files in {cache_dir}")

        # Check for zero-size files (incomplete writes)
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    if os.path.isfile(filepath) and os.path.getsize(filepath) == 0:
                        issues.append(f"Found empty file: {filepath}")
                except OSError:
                    # File might have been deleted or inaccessible
                    continue

    return len(issues) == 0, issues


def clear_cache(logger=None) -> None:
    """
    Clear all SDXL cache directories

    Args:
        logger: Optional logger for output
    """
    for cache_dir in get_cache_dirs():
        if os.path.exists(cache_dir):
            if logger:
                logger.info(f"Clearing cache: {cache_dir}")
            else:
                print(f"Clearing cache: {cache_dir}")

            try:
                shutil.rmtree(cache_dir)
                if logger:
                    logger.info(f"  ✓ Cleared {cache_dir}")
                else:
                    print(f"  ✓ Cleared {cache_dir}")
            except Exception as e:
                if logger:
                    logger.error(f"  ✗ Failed to clear {cache_dir}: {e}")
                else:
                    print(f"  ✗ Failed to clear {cache_dir}: {e}")


def log_cache_info(logger) -> None:
    """
    Log information about cache directories

    Args:
        logger: Logger instance
    """
    logger.info("Cache directories:")
    for cache_dir in get_cache_dirs():
        if os.path.exists(cache_dir):
            try:
                # Get directory size
                total_size = 0
                file_count = 0
                for root, dirs, files in os.walk(cache_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(filepath)
                            file_count += 1
                        except OSError:
                            continue

                size_mb = total_size / (1024 * 1024)
                logger.info(f"  {cache_dir}: {file_count} files, {size_mb:.1f} MB")
            except Exception as e:
                logger.info(f"  {cache_dir}: (error reading: {e})")
        else:
            logger.info(f"  {cache_dir}: (not created yet)")
