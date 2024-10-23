from os.path import join
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from docs.source.conf import doc_version


def generate_doc_container(file: Path, alias: str, includes: List[Path]):
    """
    Generates source files index.
    The generated file contains a toctree of included files.
    The resulting file is marked 'orphan' not to emit warning if not included in any toctree.
    It is also has maximum depth of 1 (only filenames) and includes titles only.

    :param file: Path to directory index file (file name will be prefixed 'index_').
    :param alias: Module name alias.
    :param includes: List of the files to include into the directory, should be sorted previously.
    """
    sources = "\n   ".join(str(include.stem) for include in includes)
    contents = f""":orphan:

.. This is an auto-generated RST file representing documentation source directory structure

{alias}
{"=" * len(alias)}

.. autosummary::
   :toctree:

   {sources}
"""
    file.with_name(f"index_{file.name}").write_text(contents)

# TODO: add a configs dictionary with all parameters inside it for ease of use.
# def regenerate_apiref(paths: Optional[List[Tuple[str, str]]] = None, root_dir: str = ".", apiref_destination: str = "apiref"):
def regenerate_apiref(paths: Optional[List[Tuple[str, str]]] = None, configs: dict = {}):
    """
    Regenerate files in apiref root.
    Not all the files there are generally useful: mostly the folder consists of module toctrees that look ugly.
    The purpose of this function is removing all sources that represent a module
    and create convenient indexes for the remaining files.
    The function also adds a special 'source_name' meta variable to every file, containing path to its source on GitHub.
    This special variable will be used by 'View on GitHub' button on documentation pages.

    :param paths: Paths to the modules that should be merged together, separated by '.'.
    Should be prefixes of files in apiref root.
    :param configs: A dictionary that contains apiref root path (default: apiref),
    doc_version and the root directory of the project.
    """
    paths = list() if paths is None else paths
    source = Path(configs.root_dir) / "docs" / "source" / configs.apiref_destination
    doc_containers: Dict[str, Tuple[str, List[Path]]] = dict()

    for doc_file in iter(source.glob("./*.rst")):
        contents = doc_file.read_text()
        if ".. toctree::" in contents:
            doc_file.unlink()
            continue

        container = next((alias for flat, alias in paths if doc_file.name.startswith(flat)), None)
        if container is None:
            doc_file.unlink()
            continue
        else:
            filename = container.replace(" ", "_").lower()
            doc_containers[filename] = container, doc_containers.get(filename, ("", list()))[1] + [doc_file]

        with open(doc_file, "r+") as file:
            contents = file.read()
            doc_file.write_text(f":source_name: {join(*doc_file.stem.split('.'))}\n"
                                f":doc_version: {configs.doc_version}\n\n{contents}")

    for name, (alias, files) in doc_containers.items():
        generate_doc_container(source / Path(f"{name}.rst"), alias, files)
