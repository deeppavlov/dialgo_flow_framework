"""
Benchmark Streamlit
-------------------
This module provides a streamlit interface to view benchmark results.
The interface has 4 tabs:

- "Benchmark sets" -- Here you can add or remove benchmark set files (generated by `save_results_to_file` function).
- "View" -- Here you can view various stats for each benchmark in the set (as well as add benchmarks to compare tab).
- "Compare" -- Here you can compare benchmarks added via "View" tab.
- "Mass compare" -- This tab lets you compare all benchmarks in a particular set.

Run this file with `streamlit run {path/to/this/file}`.

When run, this module will search for a file named "benchmark_results_files.json" in the directory
where the command above was executed.
If the file does not exist there, it will be created.
The file is used to store paths to benchmark result files.

Benchmark result files added via this module are not changed (only read).
"""
import json
import pathlib
from pathlib import Path
from uuid import uuid4
from copy import deepcopy

import pandas as pd
from pympler import asizeof
from humanize import naturalsize
import altair as alt
import streamlit as st

try:
    from benchmark_new_format import update_benchmark_file
except ImportError:
    update_benchmark_file = None


st.set_page_config(
    page_title="DB benchmark",
    layout="wide",
    initial_sidebar_state="expanded",
)

BENCHMARK_RESULTS_FILES = Path("benchmark_results_files.json")
# This file stores links to benchmark set files generated by `save_results_to_file`.

UPLOAD_FILES_DIR = Path("uploaded_benchmarks")
# This directory stores all the benchmarks uploaded via the streamlit interface

UPLOAD_FILES_DIR.mkdir(exist_ok=True)

MERGE_FILES_DIR = Path("merged_benchmarks")
# This directory stores all the benchmarks merged via the streamlit interface

MERGE_FILES_DIR.mkdir(exist_ok=True)


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


if not check_password():
    st.stop()


if not BENCHMARK_RESULTS_FILES.exists():
    with open(BENCHMARK_RESULTS_FILES, "w", encoding="utf-8") as fd:
        json.dump([], fd)

if "benchmark_files" not in st.session_state:
    with open(BENCHMARK_RESULTS_FILES, "r", encoding="utf-8") as fd:
        st.session_state["benchmark_files"] = json.load(fd)

if "benchmarks" not in st.session_state:
    st.session_state["benchmarks"] = {}

    for file in st.session_state["benchmark_files"]:
        if update_benchmark_file is not None:
            update_benchmark_file(file)
        with open(file, "r", encoding="utf-8") as fd:
            st.session_state["benchmarks"][file] = json.load(fd)

if "compare" not in st.session_state:
    st.session_state["compare"] = []


def get_diff(last_metric, first_metric):
    if last_metric is None or first_metric is None:
        return "-"
    if st.session_state["percent_compare"]:
        return f"{(last_metric / first_metric - 1):.3%}"
    else:
        return f"{last_metric - first_metric:.3}"


def add_metrics(container, value_benchmark, diff_benchmark=None, one_column=None):
    column_names = ("write", "read", "update", "read+update")

    if not value_benchmark["success"]:
        values = {key: "-" for key in column_names}
        diffs = None
    else:
        values = {key: value_benchmark["average_results"][f"pretty_{key}"] for key in column_names}

        if diff_benchmark is not None:
            if not diff_benchmark["success"]:
                diffs = {key: "-" for key in column_names}
            else:
                diffs = {
                    key: get_diff(
                        value_benchmark["average_results"][f"pretty_{key}"],
                        diff_benchmark["average_results"][f"pretty_{key}"]
                    ) for key in column_names
                }
        else:
            diffs = None

    metric_help = {
        "write": "Average write time for a context with from_dialog_len turns into a clean context storage.",
        "read": "Average read time (dialog_len ranges between from_dialog_len and to_dialog_len).",
        "update": "Average update time (dialog_len ranges between from_dialog_len and to_dialog_len).",
        "read+update": "Sum of average read and update times."
                       " This metric is the time context_storage interface takes during each of the dialog turns."
    }

    if not one_column:
        write, read, update, read_update = container.columns(4)

        columns = {
            "write": write,
            "read": read,
            "update": update,
            "read+update": read_update,
        }

        for column_name, column in columns.items():
            column.metric(
                column_name.title(),
                values[column_name],
                delta=diffs[column_name] if diffs else None,
                delta_color="inverse",
                help=metric_help[column_name]
            )
    else:
        container.metric(
            one_column.title(),
            values[one_column],
            delta=diffs[one_column] if diffs else None,
            delta_color="inverse",
            help=metric_help[one_column]
        )


def get_opposite_benchmarks(benchmark_set, benchmark):
    compare_params = (
        ("db_factory", "uri"),
        ("benchmark_config", "context_num"),
        ("benchmark_config", "from_dialog_len"),
        ("benchmark_config", "to_dialog_len"),
        ("benchmark_config", "step_dialog_len"),
        ("benchmark_config", "message_dimensions"),
        ("benchmark_config", "misc_dimensions"),
    )

    def get_param(bench, param):
        if len(param) == 1:
            return bench.get(param[0])
        else:
            return get_param(bench.get(param[0]), param[1:])

    opposite_benchmarks = [
        opposite_benchmark
        for opposite_benchmark in benchmark_set["benchmarks"]
        if opposite_benchmark["uuid"] != benchmark["uuid"] and all(
            get_param(benchmark, param) == get_param(opposite_benchmark, param) for param in compare_params
        )
    ]

    return opposite_benchmarks


def _add_benchmark(benchmark_file, container):
    benchmark_file = str(benchmark_file)

    if benchmark_file == "":
        return

    if benchmark_file in st.session_state["benchmark_files"]:
        container.warning(f"Benchmark file already added: {benchmark_file}")
        return

    if not Path(benchmark_file).exists():
        container.warning(f"File does not exists: {benchmark_file}")
        return

    if update_benchmark_file is not None:
        update_benchmark_file(benchmark_file)

    with open(benchmark_file, "r", encoding="utf-8") as fd:
        file_contents = json.load(fd)

    for benchmark in st.session_state["benchmarks"].values():
        if file_contents["uuid"] == benchmark["uuid"]:
            container.warning(f"Benchmark with the same uuid already exists: {benchmark_file}")
            return

    st.session_state["benchmark_files"].append(benchmark_file)
    with open(BENCHMARK_RESULTS_FILES, "w", encoding="utf-8") as fd:
        json.dump(list(st.session_state["benchmark_files"]), fd)
    st.session_state["benchmarks"][benchmark_file] = file_contents

    container.text(f"Added {benchmark_file}")


st.sidebar.text(f"Benchmarks take {naturalsize(asizeof.asizeof(st.session_state['benchmarks']))} RAM")

st.sidebar.divider()

st.sidebar.checkbox("Compare dev and partial in view tab", value=True, key="partial_compare_checkbox")
st.sidebar.checkbox("Percent comparison", value=True, key="percent_compare")

mass_compare_tab, add_tab, merge_tab, view_tab, compare_tab = st.tabs(["Mass compare", "Benchmark sets", "Merge", "View", "Compare"])


###############################################################################
# Benchmark file manipulation tab
# Allows adding and deleting benchmark files
###############################################################################

with add_tab:
    benchmark_list = []

    for file, benchmark_set in st.session_state["benchmarks"].items():
        benchmark_list.append(
            {
                "file": file,
                "name": benchmark_set["name"],
                "description": benchmark_set["description"],
                "uuid": benchmark_set["uuid"],
                "delete": False,
            }
        )

    benchmark_list_df = pd.DataFrame(data=benchmark_list)

    df_container = st.container()

    def edit_name_desc():
        edited_rows = st.session_state["result_df"]["edited_rows"]

        for row, edits in edited_rows.items():
            for column, column_value in edits.items():
                if column in ("name", "description"):
                    edited_file = benchmark_list_df.iat[row, 0]
                    st.session_state["benchmarks"][edited_file][column] = column_value

                    with open(edited_file, "w", encoding="utf-8") as edited_fd:
                        json.dump(st.session_state["benchmarks"][edited_file], edited_fd)

                    df_container.text(f"row {row}: changed {column} to '{column_value}'")

    edited_df = df_container.data_editor(
        benchmark_list_df,
        key="result_df",
        disabled=("file", "uuid"),
        on_change=edit_name_desc
    )

    delist_container = st.container()
    delist_container.divider()

    def delist_benchmarks():
        delisted_sets = [
            f"{name} ({uuid})" for name, uuid in edited_df.loc[edited_df["delete"]][["name", "uuid"]].values
        ]

        st.session_state["compare"] = [
            item for item in st.session_state["compare"] if item["benchmark_set"] not in delisted_sets
        ]

        files_to_delist = edited_df.loc[edited_df["delete"]]["file"]
        st.session_state["benchmark_files"] = list(set(st.session_state["benchmark_files"]) - set(files_to_delist))
        for file in files_to_delist:
            del st.session_state["benchmarks"][file]
            delist_container.text(f"Delisted {file}")

        with open(BENCHMARK_RESULTS_FILES, "w", encoding="utf-8") as fd:
            json.dump(list(st.session_state["benchmark_files"]), fd)

    delist_container.button(label="Delist selected benchmark sets", on_click=delist_benchmarks)

    st.divider()

    add_container, add_from_dir_container = st.columns(2)

    add_container.text_input(label="Benchmark set file", key="add_benchmark_file")

    def add_benchmark():
        _add_benchmark(st.session_state["add_benchmark_file"], add_container)

    add_container.button("Add one file from file", on_click=add_benchmark)

    add_from_dir_container.text_input(label="Directory with benchmark files", key="add_from_dir")

    def add_from_dir():
        dir_path = pathlib.Path(st.session_state["add_from_dir"])
        if dir_path.is_dir():
            for file in dir_path.iterdir():
                _add_benchmark(file, add_from_dir_container)

    add_from_dir_container.button("Add all files from directory", on_click=add_from_dir)

    st.divider()

    upload_container = st.container()

    def process_uploaded_files():
        uploaded_files = st.session_state["benchmark_file_uploader"]
        if uploaded_files is not None:
            if len(uploaded_files) > 0:
                new_uploaded_file_dir = UPLOAD_FILES_DIR / str(uuid4())
                new_uploaded_file_dir.mkdir()

                for file in uploaded_files:
                    file_path = new_uploaded_file_dir / file.name
                    with open(file_path, "wb") as uploaded_file_descriptor:
                        uploaded_file_descriptor.write(file.read())

                    _add_benchmark(file_path, upload_container)

    with upload_container.form("upload_form", clear_on_submit=True):
        st.file_uploader(
            "Upload benchmark results", accept_multiple_files=True, type="json", key="benchmark_file_uploader"
        )
        st.form_submit_button("Submit", on_click=process_uploaded_files)


###############################################################################
# Merge tab
# Allows merging several benchmarks from different files into a single one
###############################################################################

def get_subsets(benchmark_set):
    benchmark_subsets = []
    for benchmark in benchmark_set["benchmarks"]:
        if benchmark["name"].startswith("default"):
            benchmark_subsets.append(benchmark["name"].removeprefix("default"))

    return benchmark_subsets


with merge_tab:
    sets = {
        f"{benchmark['name']} ({benchmark['uuid']})": benchmark for benchmark in st.session_state["benchmarks"].values()
    }

    merge_tab_subsets = {}

    for set_name, benchmark_set in sets.items():
        set_subsets = get_subsets(benchmark_set)

        for set_subset in set_subsets:
            merge_tab_subsets[f"{set_name} / {set_subset}"] = benchmark_set, set_subset

    with st.empty():
        merge_container = st.container()

    def merge_subsets():
        merged_benchmark_set = {
            "name": st.session_state["merged_name"],
            "description": st.session_state["merged_desc"],
            "uuid": str(uuid4()),
            "benchmarks": []
        }
        for subset in st.session_state["merged_subsets"]["added_rows"]:
            current_set, set_subset = merge_tab_subsets[subset["subset"]]

            for potential_benchmark in current_set["benchmarks"]:
                if potential_benchmark["name"].endswith(set_subset):
                    new_benchmark = deepcopy(potential_benchmark)

                    asname = subset.get("asname", set_subset)
                    if asname is not None:
                        new_benchmark["name"] = potential_benchmark["name"].removesuffix(set_subset) + asname

                    merged_benchmark_set["benchmarks"].append(new_benchmark)

        merged_benchmark_file = (MERGE_FILES_DIR / merged_benchmark_set["uuid"]).with_suffix(".json")

        with open(merged_benchmark_file, "w", encoding="utf-8") as merged_file:
            json.dump(merged_benchmark_set, merged_file)

        _add_benchmark(merged_benchmark_file, merge_container)


    with st.form("merge_form", clear_on_submit=True):
        merge_df = st.data_editor(
            pd.DataFrame({"subset": [], "asname": []}, dtype=str),
            key="merged_subsets",
            num_rows="dynamic",
            column_config={
                "subset": st.column_config.SelectboxColumn(
                    "Benchmark subset",
                    help="Subset of a set to add",
                    width="large",
                    options=merge_tab_subsets.keys()
                ),
                "asname": st.column_config.TextColumn(
                    "Name of the subset in the resulting file",
                    help="Leave None if name should be unchanged."
                )
            }
        )

        name, desc, confirm = st.columns([2, 2, 1])

        name.text_input("Merged set name", key="merged_name")
        desc.text_input("Merged set description", key="merged_desc")
        confirm.form_submit_button("Merge", on_click=merge_subsets)


###############################################################################
# View tab
# Allows viewing existing benchmarks
###############################################################################

with view_tab:
    set_choice, benchmark_choice, compare = st.columns([3, 3, 1])

    sets = {
        f"{benchmark['name']} ({benchmark['uuid']})": benchmark for benchmark in st.session_state["benchmarks"].values()
    }
    benchmark_set = set_choice.selectbox("Benchmark set", sets.keys())

    if benchmark_set is None:
        set_choice.warning("No benchmark sets available")
        st.stop()

    selected_set = sets[benchmark_set]

    set_choice.text("Set description:")
    set_choice.markdown(selected_set["description"])

    benchmarks = {f"{benchmark['name']} ({benchmark['uuid']})": benchmark for benchmark in selected_set["benchmarks"]}

    benchmark = benchmark_choice.selectbox("Benchmark", benchmarks.keys())

    if benchmark is None:
        benchmark_choice.warning("No benchmarks in the set")
        st.stop()

    selected_benchmark = benchmarks[benchmark]

    benchmark_choice.text("Benchmark description:")
    benchmark_choice.markdown(selected_benchmark["description"])

    with st.expander("Benchmark stats"):
        reproducible_stats = {
            stat: selected_benchmark[stat]
            for stat in (
                "db_factory",
                "benchmark_config",
            )
        }

        size_stats = {stat: naturalsize(value, gnu=True) for stat, value in selected_benchmark["sizes"].items()}

        st.json(reproducible_stats)
        st.json(size_stats)

    if not selected_benchmark["success"]:
        st.warning(selected_benchmark["result"])
    else:
        opposite_benchmark = None

        if st.session_state["partial_compare_checkbox"]:
            opposite_benchmarks = get_opposite_benchmarks(selected_set, selected_benchmark)

            if len(opposite_benchmarks) == 1:
                opposite_benchmark = opposite_benchmarks[0]

        add_metrics(st.container(), selected_benchmark, opposite_benchmark)

        if opposite_benchmark is not None:
            st.text(f"* In comparison with {opposite_benchmark['name']} ({opposite_benchmark['uuid']})")

        compare_item = {
            "benchmark_set": benchmark_set,
            "benchmark": benchmark,
            "write": selected_benchmark["average_results"]["pretty_write"],
            "read": selected_benchmark["average_results"]["pretty_read"],
            "update": selected_benchmark["average_results"]["pretty_update"],
            "read+update": selected_benchmark["average_results"]["pretty_read+update"],
        }

        def add_results_to_compare_tab():
            if compare_item not in st.session_state["compare"]:
                st.session_state["compare"].append(compare_item)
            else:
                st.session_state["compare"].remove(compare_item)

        compare.button(
            "Add to Compare" if compare_item not in st.session_state["compare"] else "Remove from Compare",
            on_click=add_results_to_compare_tab,
        )

        select_graph, graph = st.columns([1, 3])

        average_results = selected_benchmark["average_results"]

        graphs = {
            "Write": selected_benchmark["result"]["write_times"],
            "Read (grouped by contex_num)": average_results["read_times_grouped_by_context_num"],
            "Read (grouped by dialog_len)": average_results["read_times_grouped_by_dialog_len"],
            "Update (grouped by contex_num)": average_results["update_times_grouped_by_context_num"],
            "Update (grouped by dialog_len)": average_results["update_times_grouped_by_dialog_len"],
        }

        selected_graph = select_graph.selectbox("Select graph to display", graphs.keys())

        graph_data = graphs[selected_graph]

        if isinstance(graph_data, dict):
            data = pd.DataFrame({"dialog_len": graph_data.keys(), "time": graph_data.values()})
        else:
            data = pd.DataFrame({"context_num": range(len(graph_data)), "time": graph_data})

        chart = (
            alt.Chart(data)
            .mark_circle()
            .encode(
                x=alt.X(
                    "dialog_len:Q" if isinstance(graph_data, dict) else "context_num:Q", scale=alt.Scale(zero=False)
                ),
                y="time:Q",
            )
            .interactive()
        )

        graph.altair_chart(chart, use_container_width=True)


###############################################################################
# Compare tab
# Allows viewing existing benchmarks
###############################################################################

with compare_tab:
    df = pd.DataFrame(st.session_state["compare"])

    if not df.empty:
        st.dataframe(
            df.style.highlight_min(
                axis=0, subset=["write", "read", "update", "read+update"], props="background-color:green;"
            ).highlight_max(axis=0, subset=["write", "read", "update", "read+update"], props="background-color:red;")
        )

###############################################################################
# Mass compare tab
# Allows comparing all benchmarks inside a single set
###############################################################################

with mass_compare_tab:
    select_box_column, compact_column, link_column = st.columns([6, 2, 1])

    sets = {
        f"{benchmark_set['name']} ({benchmark_set['uuid']})": benchmark_set
        for benchmark_set in st.session_state["benchmarks"].values()
    }

    set_indexes = {
        benchmark_set['uuid']: index
        for index, benchmark_set in enumerate(st.session_state["benchmarks"].values())
    }

    modes = ("all", "read", "write", "update", "read+update")

    params = st.experimental_get_query_params()

    queried_set = params.get("mass_compare_set", [])
    set_index = 0
    if len(queried_set) == 1:
        set_index = set_indexes.get(queried_set[0], 0)
    queried_mode = params.get("metric", [])
    mode_index = 4
    if len(queried_mode) == 1:
        mode_index = int(queried_mode[0])

    benchmark_set = select_box_column.selectbox("Benchmark set", sets.keys(), key="mass_compare_selectbox", index=set_index)

    if benchmark_set is None:
        st.warning("No benchmark sets available")
        st.stop()

    selected_mode = compact_column.selectbox("Metrics to display", modes, index=mode_index)

    link_column.markdown(f"[Link](?mass_compare_set={benchmark_set.rsplit('(', maxsplit=1)[1].removesuffix(')')}&metric={modes.index(selected_mode)})")

    selected_set = sets[benchmark_set]

    added_benchmarks = set()

    benchmark_clusters = []

    for benchmark in selected_set["benchmarks"]:
        if benchmark["uuid"] in added_benchmarks:
            continue

        opposite_benchmarks = get_opposite_benchmarks(selected_set, benchmark)

        added_benchmarks.add(benchmark["uuid"])
        added_benchmarks.update({bm["uuid"] for bm in opposite_benchmarks})
        benchmark_clusters.append([benchmark, *opposite_benchmarks])

    if selected_mode == "all":
        for benchmark_cluster in benchmark_clusters:
            st.divider()

            benchmark, *opposite_benchmarks = benchmark_cluster

            st.subheader(f"{benchmark['name']} ({benchmark['uuid']})")
            add_metrics(st.container(), benchmark)

            last_benchmark = benchmark

            for opposite_benchmark in opposite_benchmarks:
                st.subheader(f"{opposite_benchmark['name']} ({opposite_benchmark['uuid']})")
                add_metrics(st.container(), opposite_benchmark, last_benchmark)
                last_benchmark = opposite_benchmark
    else:
        subsets = get_subsets(selected_set)

        for benchmark_cluster in benchmark_clusters:
            if not all([benchmark["name"].endswith(subset) for benchmark, subset in zip(benchmark_cluster, subsets)]):
                st.warning("Benchmarks with the same configs have different set names")
                st.stop()

        if not all([len(cluster) == len(benchmark_clusters[0]) for cluster in benchmark_clusters]):
            st.warning("Benchmarks with the same configs have different lengths")
            st.stop()

        configs = []
        for benchmark_cluster in benchmark_clusters:
            if not benchmark_cluster[0]["name"].endswith(subsets[0]):
                st.warning(f"First benchmark is not from {subsets[0]}")
                st.stop()
            config_name = benchmark_cluster[0]["name"].removesuffix(subsets[0])
            configs.append((config_name, benchmark_cluster[0]["benchmark_config"]))

        st.divider()
        _, *config_columns = st.columns(len(configs) + 1)

        for config, config_column in zip(configs, config_columns):
            config_column.markdown(config[0], help=str(config[1]))

        for index, subset in enumerate(subsets):
            st.divider()
            subset_column, *metric_columns = st.columns(len(configs) + 1)

            subset_column.markdown(subset)

            for benchmark_cluster, metric_column in zip(benchmark_clusters, metric_columns):
                if index == 0:
                    add_metrics(metric_column, benchmark_cluster[index], one_column=selected_mode)
                else:
                    add_metrics(metric_column, benchmark_cluster[index], benchmark_cluster[index - 1], one_column=selected_mode)
