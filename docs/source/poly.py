from pathlib import Path
from datetime import datetime
from functools import partial
from sphinx_polyversion import *
from sphinx_polyversion.git import *
from sphinx_polyversion.git import closest_tag
from sphinx_polyversion.pyvenv import Poetry
from docs.source.builder import ChatskySphinxBuilder
from docs.source.switcher_gen import generate_switcher
import git
import os

# Generate switcher.json file
generate_switcher()

# Regex matching the branches to build docs for
# This regex stands for all branches except master, so docs can be built for any branch on demand.
# (if the workflow is launched from it)
BRANCH_REGEX = r"((?!master).)*"
# BRANCH_REGEX = r".*"

#: Regex matching the tags to build docs for
TAG_REGEX = r"-"

# This variable is set to `False` during workflow build. It is 'True' during local builds.
LOCAL_BUILD = os.getenv('LOCAL_BUILD', default="True")

repo = git.Repo('./')
# This variable is needed for passing the branch name during PR workflow doc builds,
# because in those cases 'repo.active_branch' gives 'detached HEAD'.
branch = os.getenv('BRANCH_NAME', default=None)
print(branch)
print(repo)
if branch is None:
    branch = repo.active_branch

if LOCAL_BUILD == "True":
    # Local builds only build docs for the current branch and no tags.
    BRANCH_REGEX = str(branch)
    TAG_REGEX = r"-"
elif str(branch) == "master":
    # Releases are handled here (pushes into master mean a release, so the latest tag is built)
    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime)
    latest_tag = tags[-1]
    TAG_REGEX = str(latest_tag)

#: Output dir relative to project root
OUTPUT_DIR = "docs/build"

#: Source directory
SOURCE_DIR = "docs/source"

#: Arguments to pass to `poetry install`
POETRY_ARGS = "--with tutorials,docs --all-extras --no-ansi --no-interaction".split()

#: Arguments to pass to `sphinx-build`
SPHINX_ARGS = "-b html -W --keep-going -v".split()

#: Mock data used for building local version
MOCK_DATA = {
    "revisions": [
        GitRef("dev", "", "", GitRefType.BRANCH, datetime.fromtimestamp(0)),
        GitRef("master", "", "", GitRefType.BRANCH, datetime.fromtimestamp(1)),
    ],
    "current": GitRef("local", "", "", GitRefType.BRANCH, datetime.fromtimestamp(2)),
}
MOCK = False

# Load overrides read from commandline to global scope
apply_overrides(globals())

# Determine repository root directory
root = Git.root(Path(__file__).parent)
src = Path(SOURCE_DIR)

# Setup driver and run it
DefaultDriver(
    root,
    OUTPUT_DIR,
    vcs=Git(
        branch_regex=BRANCH_REGEX,
        tag_regex=TAG_REGEX,
        buffer_size=1 * 10**9,  # 1 GB
        predicate=file_predicate([src]), # exclude refs without source dir
    ),
    builder=ChatskySphinxBuilder(src, args=SPHINX_ARGS),
    env=Poetry.factory(args=POETRY_ARGS),
    selector=partial(closest_tag, root),
    template_dir=root / src / "templates",
    static_dir=root / src / "static",
    mock=MOCK_DATA,
).run(MOCK)
