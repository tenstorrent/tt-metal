import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)


def pytest_configure(config):
    config.addinivalue_line("markers", "numerics: golden regression tests (test_golden_grad.py)")
