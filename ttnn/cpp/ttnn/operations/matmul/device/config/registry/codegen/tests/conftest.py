# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest


@pytest.fixture
def expect_error():
    """Keep the standalone emitter suite aligned with the repository fixture."""

    @contextlib.contextmanager
    def expect_error_(error, message):
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: implements expect_error
            yield exc_info

    return expect_error_
