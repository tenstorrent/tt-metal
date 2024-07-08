# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from setuptools import find_packages

setup(
    name="ttnn",
    version="0.0.0",
    description="User-friendly API for running operations on TensTorrent hardware",
    packages=find_packages(),
)
