#  Copyright (c) 2024.
#  Licensed under the MIT license
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--render",
        action="store_true",
        default=False,
        help="Run tests with rendering (render_mode=human).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "render: mark tests as using rendering.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--render"):
        # do not skip
        return
    skip_render = pytest.mark.skip(reason="need --render option to run")
    for item in items:
        if "render" in item.keywords:
            item.add_marker(skip_render)
