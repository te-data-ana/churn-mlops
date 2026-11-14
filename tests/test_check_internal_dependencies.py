import runpy
from itertools import combinations
from math import hypot
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import pytest

SCRIPT_API = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts" / "check_internal_dependencies.py")
)
build_plotly_graph = SCRIPT_API["build_plotly_graph"]
build_plotly_dependency_matrix = SCRIPT_API["build_plotly_dependency_matrix"]
separate_cluster_centers = SCRIPT_API["_separate_cluster_centers"]
write_dependency_visualizations = SCRIPT_API["write_dependency_visualizations"]


def test_build_plotly_graph_shows_dependency_direction_and_submodule_colors() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("churn_mlops.config.loader", "churn_mlops.models.catalog"),
            ("churn_mlops.models.catalog", "churn_mlops.tracking.registry"),
            ("churn_mlops.tracking.registry", "churn_mlops.models.catalog"),
            ("churn_mlops.data.storage", "churn_mlops.data.ingestion"),
        ]
    )
    graph.add_nodes_from(
        [
            "churn_mlops.data.storage",
            "churn_mlops.evaluation.metrics",
            "churn_mlops.monitoring.core",
            "churn_mlops.serving.api",
            "churn_mlops.training.tuning",
            "churn_mlops.train",
        ]
    )

    figure = build_plotly_graph(graph)

    node_traces = [trace for trace in figure.data if trace.mode == "markers+text"]
    expected_groups = {
        "Config",
        "Data",
        "Evaluation",
        "Models",
        "Monitoring",
        "Serving",
        "Tracking",
        "Training",
        "Root-level",
    }

    assert {trace.name for trace in node_traces} == expected_groups
    assert len({trace.marker.color for trace in node_traces}) == len(expected_groups)
    labels = {trace.name: set(trace.text) for trace in node_traces}
    assert "config.loader" in labels["Config"]
    assert "models.catalog" in labels["Models"]
    assert "data.ingestion" in labels["Data"]
    assert "train" in labels["Root-level"]
    data_trace = next(trace for trace in node_traces if trace.name == "Data")
    data_node_distance = (
        (data_trace.x[0] - data_trace.x[1]) ** 2
        + (data_trace.y[0] - data_trace.y[1]) ** 2
    ) ** 0.5
    assert data_node_distance > 4.5
    assert len(figure.layout.annotations) == graph.number_of_edges()
    assert all(annotation.showarrow for annotation in figure.layout.annotations)
    assert any(trace.name == "In dependency cycle" for trace in figure.data)


def test_build_plotly_dependency_matrix_counts_directional_module_dependencies() -> (
    None
):
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("churn_mlops.config.loader", "churn_mlops.data.ingestion"),
            ("churn_mlops.data.storage", "churn_mlops.data.ingestion"),
            ("churn_mlops.models.catalog", "churn_mlops.tracking.registry"),
            ("churn_mlops.tracking.registry", "churn_mlops.models.catalog"),
        ]
    )

    figure = build_plotly_dependency_matrix(graph)
    matrix = figure.data[0]

    assert tuple(matrix.x) == ("config", "data", "models", "tracking")
    assert tuple(matrix.y) == ("config", "data", "models", "tracking")
    assert tuple(tuple(row) for row in matrix.z) == (
        (0, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
    )
    assert "config.loader -> data.ingestion" in matrix.hovertext[0][1]
    assert "No direct module dependencies" in matrix.hovertext[0][0]
    assert figure.layout.xaxis.title.text == "Imported dependency"
    assert figure.layout.yaxis.title.text == "Importing submodule"


def test_write_dependency_visualizations_creates_and_uses_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = nx.DiGraph([("churn_mlops.config.loader", "churn_mlops.data.ingestion")])
    graph_figure = build_plotly_graph(graph)
    matrix_figure = build_plotly_dependency_matrix(graph)
    written_paths: list[Path] = []

    def record_write_html(_figure: go.Figure, path: str) -> None:
        written_paths.append(Path(path))

    monkeypatch.setattr(type(graph_figure), "write_html", record_write_html)
    output_dir = tmp_path / "output"

    graph_path, matrix_path = write_dependency_visualizations(
        graph_figure,
        matrix_figure,
        output_dir,
    )

    assert output_dir.is_dir()
    assert graph_path == output_dir / "dependency_graph.html"
    assert matrix_path == output_dir / "dependency_matrix.html"
    assert written_paths == [graph_path, matrix_path]


def test_separate_cluster_centers_reserves_space_by_cluster_radius() -> None:
    initial_positions = {
        "small": (0.0, 0.0),
        "large": (1.0, 0.0),
        "medium": (2.0, 0.0),
    }
    radii = {"small": 1.0, "large": 3.0, "medium": 1.5}

    positions = separate_cluster_centers(initial_positions, radii)

    for group_a, group_b in combinations(positions, 2):
        distance = hypot(
            positions[group_a][0] - positions[group_b][0],
            positions[group_a][1] - positions[group_b][1],
        )
        assert distance >= radii[group_a] + radii[group_b] - 1e-6
