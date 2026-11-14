from __future__ import annotations

import ast
from itertools import combinations
from math import hypot, sqrt
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go

SUBMODULE_COLORS = {
    "config": "#4E79A7",
    "data": "#F28E2B",
    "evaluation": "#E15759",
    "models": "#76B7B2",
    "monitoring": "#59A14F",
    "serving": "#EDC949",
    "tracking": "#AF7AA1",
    "training": "#FF9DA7",
    "root": "#9D9D9D",
}


def _get_node_group(node: str) -> str:
    module_parts = node.split(".")
    if len(module_parts) > 1 and module_parts[1] in SUBMODULE_COLORS:
        return module_parts[1]
    return "root"


def _get_node_label(node: str) -> str:
    module_parts = node.split(".")
    if len(module_parts) > 1:
        return ".".join(module_parts[1:])
    return node


def _separate_cluster_centers(
    positions: dict[str, tuple[float, float]],
    radii: dict[str, float],
) -> dict[str, tuple[float, float]]:
    """
    Separates cluster centers to avoid overlap.

    Args:
        positions: A dictionary mapping cluster names to their initial positions.
        radii: A dictionary mapping cluster names to their radii.

    Returns:
        A dictionary mapping cluster names to their separated positions.
    """
    separated_positions = positions.copy()
    groups = list(separated_positions)

    for _ in range(500):
        overlapping = False
        for group_a, group_b in combinations(groups, 2):
            x_a, y_a = separated_positions[group_a]
            x_b, y_b = separated_positions[group_b]
            dx = x_b - x_a
            dy = y_b - y_a
            distance = hypot(dx, dy)
            minimum_distance = radii[group_a] + radii[group_b]

            if distance >= minimum_distance - 1e-6:
                continue

            if distance < 1e-9:
                unit_x, unit_y = 1.0, 0.0
                overlap = minimum_distance
            else:
                unit_x = dx / distance
                unit_y = dy / distance
                overlap = minimum_distance - distance

            shift = overlap / 2
            separated_positions[group_a] = (
                x_a - unit_x * shift,
                y_a - unit_y * shift,
            )
            separated_positions[group_b] = (
                x_b + unit_x * shift,
                y_b + unit_y * shift,
            )
            overlapping = True

        if not overlapping:
            return separated_positions

    raise RuntimeError("Unable to separate dependency graph clusters")


def discover_modules(package_root: Path) -> dict[str, Path]:
    """
    Determines all modules of a package.

    Args:
        package_root: Path to the root directory of the package.

    Returns:
        A mapping of module names to their corresponding file paths.

    Example:
        mypackage/sub/a.py
    ->  mypackage.sub.a
    """
    modules = {}

    package_name = package_root.name

    for py_file in package_root.rglob("*.py"):
        relative = py_file.relative_to(package_root)

        if py_file.name == "__init__.py":
            parts = relative.parts[:-1]
        else:
            parts = relative.with_suffix("").parts

        module_name = ".".join([package_name, *parts])

        modules[module_name] = py_file

    return modules


def extract_internal_imports(
    package_root: Path,
) -> nx.DiGraph:
    """
    Determines the internal dependencies of a package.

    Args:
        package_root: Path to the root directory of the package.

    Returns:
        nx.DiGraph: A directed graph representing the internal dependencies.
    """

    modules = discover_modules(package_root)

    G = nx.DiGraph()

    for module_name in modules:
        G.add_node(module_name)

    package_prefix = package_root.name

    for source_module, py_file in modules.items():
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError) as ex:
            print(f"Skip {py_file}: {ex}")
            continue

        for node in ast.walk(tree):
            # import xxx
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name

                    if imported.startswith(package_prefix):
                        G.add_edge(source_module, imported)

            # from xxx import yyy
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module

                if imported.startswith(package_prefix):
                    G.add_edge(source_module, imported)

    return G


def build_plotly_graph(G: nx.DiGraph) -> go.Figure:
    """
    Builds a Plotly graph from a NetworkX directed graph.

    Args:
        G: The directed graph (nx.DiGraph) to visualize.

    Returns:
        The Plotly figure (go.Figure) representing the graph.
    """

    node_groups: dict[str, list[str]] = {}
    for node in G.nodes():
        node_groups.setdefault(_get_node_group(node), []).append(node)

    group_graph = nx.Graph()
    group_graph.add_nodes_from(node_groups)
    for source, target in G.edges():
        source_group = _get_node_group(source)
        target_group = _get_node_group(target)
        if source_group != target_group:
            if group_graph.has_edge(source_group, target_group):
                group_graph[source_group][target_group]["weight"] += 1
            else:
                group_graph.add_edge(source_group, target_group, weight=1)

    group_positions = nx.spring_layout(
        group_graph,
        seed=42,
        k=3.5,
        scale=10,
    )
    local_positions_by_group = {}
    cluster_radii = {}
    for index, (group, nodes) in enumerate(node_groups.items()):
        local_positions = nx.spring_layout(
            G.subgraph(nodes),
            seed=42 + index,
            k=1.6,
            scale=3.0 * (0.5 + 0.5 * len(nodes)),
        )
        local_positions_by_group[group] = local_positions
        cluster_radii[group] = (
            max(
                (
                    hypot(float(position[0]), float(position[1]))
                    for position in local_positions.values()
                ),
                default=0.0,
            )
            + 1.0
            + 0.35 * sqrt(len(nodes))
        )

    spaced_group_positions = _separate_cluster_centers(
        {
            group: (float(position[0]), float(position[1]))
            for group, position in group_positions.items()
        },
        cluster_radii,
    )
    pos = {}
    for group, nodes in node_groups.items():
        for node, local_position in local_positions_by_group[group].items():
            pos[node] = local_position + spaced_group_positions[group]

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_arrows = []

    for source, target in G.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_arrows.append(
            {
                "x": x0 + (x1 - x0) * 0.68,
                "y": y0 + (y1 - y0) * 0.68,
                "ax": x0 + (x1 - x0) * 0.52,
                "ay": y0 + (y1 - y0) * 0.52,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "text": "",
                "showarrow": True,
                "arrowhead": 3,
                "arrowsize": 1.2,
                "arrowwidth": 1.2,
                "arrowcolor": "rgba(80, 80, 80, 0.8)",
            }
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 1},
        hoverinfo="none",
    )

    cycle_nodes = {n for cycle in nx.simple_cycles(G) for n in cycle}

    node_traces = []
    for group, color in SUBMODULE_COLORS.items():
        nodes = node_groups.get(group, [])
        if not nodes:
            continue

        node_traces.append(
            go.Scatter(
                x=[pos[node][0] for node in nodes],
                y=[pos[node][1] for node in nodes],
                mode="markers+text",
                text=[_get_node_label(node) for node in nodes],
                textposition="top center",
                textfont={"size": 10},
                hovertext=[
                    f"""
<b>{node}</b><br>
Fan-In: {G.in_degree(node)}<br>
Fan-Out: {G.out_degree(node)}<br>
Degree: {G.in_degree(node) + G.out_degree(node)}
"""
                    for node in nodes
                ],
                hoverinfo="text",
                name=group.title() if group != "root" else "Root-level",
                legendgroup=group,
                marker={
                    "size": [
                        15 + (G.in_degree(node) + G.out_degree(node)) * 3
                        for node in nodes
                    ],
                    "color": color,
                    "line": {
                        "width": 2,
                        "color": [
                            "crimson" if node in cycle_nodes else "black"
                            for node in nodes
                        ],
                    },
                },
            )
        )

    if cycle_nodes:
        node_traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="In dependency cycle",
                marker={
                    "size": 10,
                    "color": "white",
                    "line": {"width": 2, "color": "crimson"},
                },
            )
        )

    fig = go.Figure(
        data=[
            edge_trace,
            *node_traces,
        ]
    )

    fig.update_layout(
        title="Python Package Dependency Graph",
        showlegend=True,
        legend_title_text="First-level submodule",
        hovermode="closest",
        annotations=edge_arrows,
        margin={
            "l": 20,
            "r": 20,
            "t": 40,
            "b": 20,
        },
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "visible": False,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "visible": False,
        },
    )

    return fig


def build_plotly_dependency_matrix(G: nx.DiGraph) -> go.Figure:
    """
    Builds a submodule-level dependency matrix from a directed graph.

    Args:
        G: The directed graph of module dependencies.

    Returns:
        A Plotly heatmap whose rows are importing submodules and columns are
        imported dependency submodules.
    """
    groups = [
        group
        for group in SUBMODULE_COLORS
        if any(_get_node_group(node) == group for node in G.nodes())
    ]
    group_indices = {group: index for index, group in enumerate(groups)}
    dependency_counts = [[0 for _ in groups] for _ in groups]
    dependency_details: list[list[list[str]]] = [[[] for _ in groups] for _ in groups]

    for source, target in G.edges():
        source_group = _get_node_group(source)
        target_group = _get_node_group(target)
        row = group_indices[source_group]
        column = group_indices[target_group]
        dependency_counts[row][column] += 1
        dependency_details[row][column].append(
            f"{_get_node_label(source)} -> {_get_node_label(target)}"
        )

    hover_text = [
        [
            "<br>".join(dependency_details[row][column])
            or "No direct module dependencies"
            for column in range(len(groups))
        ]
        for row in range(len(groups))
    ]
    maximum_count = max(
        (max(row) for row in dependency_counts),
        default=0,
    )

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=groups,
                y=groups,
                z=dependency_counts,
                text=dependency_counts,
                texttemplate="%{text}",
                textfont={"size": 14},
                hovertext=hover_text,
                hovertemplate=(
                    "Importing submodule: %{y}<br>"
                    "Imported dependency: %{x}<br>"
                    "Module dependencies: %{z}<br>"
                    "%{hovertext}<extra></extra>"
                ),
                colorscale=[
                    [0, "#F3F4F6"],
                    [1, "#2166AC"],
                ],
                zmin=0,
                zmax=max(1, maximum_count),
                colorbar={"title": "Module dependencies"},
            )
        ]
    )
    fig.update_layout(
        title="Submodule Dependency Matrix",
        width=900,
        height=750,
        xaxis_title="Imported dependency",
        yaxis_title="Importing submodule",
        yaxis={"autorange": "reversed"},
        margin={"l": 100, "r": 40, "t": 60, "b": 100},
    )
    return fig


def write_dependency_visualizations(
    graph_figure: go.Figure,
    matrix_figure: go.Figure,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Writes both dependency visualizations into a dedicated output directory.

    Args:
        graph_figure: The Plotly dependency graph.
        matrix_figure: The Plotly submodule dependency matrix.
        output_dir: The directory for the generated HTML files.

    Returns:
        The paths to the graph and matrix HTML files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "dependency_graph.html"
    matrix_path = output_dir / "dependency_matrix.html"
    graph_figure.write_html(str(graph_path))
    matrix_figure.write_html(str(matrix_path))
    return graph_path, matrix_path


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    PACKAGE_PATH = PROJECT_ROOT / "src" / "churn_mlops"
    OUTPUT_DIR = PROJECT_ROOT / "output"

    graph = extract_internal_imports(package_root=PACKAGE_PATH)

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    fig = build_plotly_graph(graph)
    matrix = build_plotly_dependency_matrix(graph)
    graph_path, matrix_path = write_dependency_visualizations(
        fig,
        matrix,
        OUTPUT_DIR,
    )
    print(f"Graph written to {graph_path}")
    print(f"Matrix written to {matrix_path}")

    fig.show()

    matrix.show()
