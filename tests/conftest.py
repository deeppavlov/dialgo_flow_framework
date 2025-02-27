import logging

import pytest

from chatsky import Pipeline, Context, AbsoluteNodeLabel


def pytest_report_header(config, start_path):
    print(f"allow_skip: {config.getoption('--allow-skip')}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Fails tests that skip, if skipping them is not allowed via config.

    Inspired by https://github.com/jankatins/pytest-error-for-skips.
    """
    outcome = yield
    rep = outcome.get_result()

    allow_skip = item.config.getoption("--allow-skip")

    if allow_skip == "all":
        return

    test_marks = [mark.name for mark in item.own_markers]

    # check that any of the permitted marks is present
    if allow_skip != "none":
        if any([mark in test_marks for mark in allow_skip.split(",")]):
            return

    if rep.skipped and call.excinfo.errisinstance(pytest.skip.Exception):
        rep.outcome = "failed"
        r = call.excinfo._getreprcrash()
        rep.longrepr = "Forbidden skipped test - {message}".format(message=r.message)


def pytest_addoption(parser):
    parser.addoption(
        "--allow-skip",
        action="store",
        default="all",
        help="A comma-separated list of marks. Any test without a mark from the list will fail on skip."
        " If not passed, every test is permitted to skip."
        " Pass `none` to disallow any test from skipping.",
    )


@pytest.fixture
def log_event_catcher():
    """
    Return a function that takes a logger and returns a list.
    Logger will put `LogRecord` objects into the list.

    Optionally, the function accepts `level` to set minimum log level.
    """

    def inner(logger, *, level=logging.DEBUG):
        logs = []

        class Handler(logging.Handler):
            def emit(self, record) -> bool:
                logs.append(record)
                return True

        logger.addHandler(Handler())
        logger.setLevel(level)
        return logs

    return inner


@pytest.fixture
def pipeline():
    return Pipeline(
        script={"flow": {"node1": {}, "node2": {}, "node3": {}}, "service": {"start": {}, "fallback": {}}},
        start_label=("service", "start"),
        fallback_label=("service", "fallback"),
    )


@pytest.fixture
def context_factory(pipeline):
    def _context_factory(forbidden_fields=None, start_label=None):
        ctx = Context()
        if start_label is not None:
            ctx.labels[0] = AbsoluteNodeLabel.model_validate(start_label)
        ctx.framework_data.pipeline = pipeline
        if forbidden_fields is not None:

            class Forbidden:
                def __init__(self, name):
                    self.name = name

                class ForbiddenError(Exception):
                    pass

                def __getattr__(self, item):
                    raise self.ForbiddenError(f"{self.name!r} is forbidden")

            for forbidden_field in forbidden_fields:
                ctx.__setattr__(forbidden_field, Forbidden(forbidden_field))
        return ctx

    return _context_factory
