from typing import TYPE_CHECKING
import re
from pathlib import Path
import os

import pytest

if TYPE_CHECKING:
    from pytest_virtualenv import VirtualEnv

from docs.source.utils.notebook import InstallationCell

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
CHATSKY_TUTORIAL_PY_FILES = map(str, (PROJECT_ROOT_DIR / "tutorials").glob("./**/*.py"))


def check_tutorial_dependencies(venv: "VirtualEnv", tutorial_source_code: str):
    """
    Install dependencies required by a tutorial in `venv` and run the tutorial.

    :param venv: Virtual environment to run the tutorial in.
    :param tutorial_source_code: Source code of the tutorial (unmodified by `apply_replace_patterns`).
    :param tmp_path: Temporary path to save the tutorial to.
    :return:
    """
    tutorial_path = venv.workspace / "tutorial.py"

    venv.env["DISABLE_INTERACTIVE_MODE"] = "1"

    with open(tutorial_path, "w") as fd:
        fd.write(tutorial_source_code)
    for deps in re.findall(InstallationCell.pattern, tutorial_source_code):
        venv.run(
            InstallationCell.replace_versions(
                f"python -m pip install {deps[0]}".replace("=={chatsky}", "").replace("chatsky", ".")
            ),
            check_rc=True,
            cd=os.getcwd(),
        )

    venv.run(f"python {tutorial_path}", check_rc=True)


@pytest.mark.parametrize("chatsky_tutorial_py_file", CHATSKY_TUTORIAL_PY_FILES)
@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.rasa
@pytest.mark.dialogflow
@pytest.mark.huggingface
@pytest.mark.no_coverage
@pytest.mark.needs_dependencies
def test_tutorials(chatsky_tutorial_py_file, virtualenv):
    with open(chatsky_tutorial_py_file, "r", encoding="utf-8") as fd:
        source_code = fd.read()

    check_tutorial_dependencies(
        virtualenv,
        source_code,
    )
