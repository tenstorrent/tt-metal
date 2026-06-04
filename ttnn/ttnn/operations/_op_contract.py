# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Registry-model support-contract exceptions.

An op's ``validate()`` raises one of these when it refuses an input:

- ``UnsupportedAxisValue`` — an axis value outside the op's ``SUPPORTED`` set.
- ``ExcludedCell``         — a cell inside ``SUPPORTED`` refused via ``EXCLUSIONS``.

Both subclass ``NotImplementedError``, so existing ``except NotImplementedError``
handlers and the "refinement candidate" semantics are unchanged. Being *typed*
lets the eval harness recognize a deliberate support refusal by ``isinstance``
instead of matching on message wording — so the human-readable message is free
to change without breaking the xfail gate.

This module lives in ttnn (not the eval harness) precisely so the op — which
ships in ttnn and may not import the eval harness — and the harness's runtime
xfail hook can share one definition.
"""


class SupportRefusal(NotImplementedError):
    """Base: the op deliberately refuses this input under the registry contract."""


class UnsupportedAxisValue(SupportRefusal):
    """An axis value is outside the op's SUPPORTED set."""


class ExcludedCell(SupportRefusal):
    """A cell inside SUPPORTED that the op refuses via EXCLUSIONS."""
