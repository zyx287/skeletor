import numpy as np
import pandas as pd
import pytest
import trimesh

from skeletor.post.postprocessing import trim_terminal_nodes, collapse_soma_nodes
from skeletor.skeletonize.base import Skeleton
from skeletor.skeletonize import wave


def _make_swc(rows):
    return pd.DataFrame(rows, columns=['node_id', 'parent_id', 'x', 'y', 'z', 'radius', 'type'])


def test_trim_terminal_nodes_rounds_and_reindex():
    swc = _make_swc([
        [0, -1, 0, 0, 0, 1.0, 0],
        [1, 0, 1, 0, 0, 1.0, 0],
        [2, 1, 2, 0, 0, 1.0, 0],
        [3, 1, 1, 1, 0, 1.0, 0],
    ])
    skel = Skeleton(swc=swc)

    out1 = trim_terminal_nodes(skel, rounds=1, keep_roots=True, reindex=True)
    assert set(out1.swc.node_id.values) == {0, 1}

    out2 = trim_terminal_nodes(skel, rounds=2, keep_roots=True, reindex=True)
    assert set(out2.swc.node_id.values) == {0}
    assert np.all(out2.swc.parent_id.values == -1)


def test_collapse_soma_nodes_rewires_children():
    swc = _make_swc([
        [0, -1, 2.0, 0.0, 0.0, 0.2, 0],
        [1, 0, 0.1, 0.0, 0.0, 0.2, 0],
        [2, 1, 0.0, 0.1, 0.0, 0.2, 0],
        [3, 2, -0.1, 0.0, 0.0, 0.2, 0],
        [4, 3, -1.5, 0.0, 0.0, 0.2, 0],
    ])
    skel = Skeleton(swc=swc)
    soma = trimesh.creation.icosphere(subdivisions=1, radius=0.5)

    out = collapse_soma_nodes(skel, soma_mesh=soma, reindex=True)

    assert len(out.swc) == 3
    soma_rows = out.swc[out.swc['type'] == 1]
    assert len(soma_rows) == 1
    soma_id = int(soma_rows.node_id.iloc[0])

    child_row = out.swc[np.isclose(out.swc['x'].values, -1.5)]
    assert int(child_row.parent_id.iloc[0]) == soma_id


def test_by_wavefront_keep_loops_returns_edges_vertices_tuple():
    mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=8)
    edges, vertices = wave.by_wavefront_keep_loops(mesh, waves=1, progress=False)

    assert isinstance(edges, np.ndarray)
    assert isinstance(vertices, np.ndarray)
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert vertices.ndim == 2 and vertices.shape[1] == 3


def test_by_wavefront_keep_loops_keeps_cycle_edges(monkeypatch):
    edges_cycle = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
    node_centers = np.array([[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]])
    node_radii = np.array([1.0, 1.0, 1.0], dtype=float)
    mesh_map = np.array([0, 1, 2], dtype=int)
    G_cycle = wave.ig.Graph(edges=edges_cycle, directed=False)
    mesh_stub = type('MeshStub', (), {
        'vertices': node_centers,
        'faces': np.array([[0, 1, 2]], dtype=int),
    })()

    monkeypatch.setattr(wave, '_wavefront_contracted_graph',
                        lambda **kwargs: (node_centers, node_radii, G_cycle.copy(), mesh_map))
    monkeypatch.setattr(wave, 'make_trimesh', lambda mesh, validate=False: mesh)

    edges, vertices = wave.by_wavefront_keep_loops(mesh_stub, waves=1, progress=False)
    expected = {tuple(sorted(e)) for e in edges_cycle.tolist()}
    observed = {tuple(sorted(e)) for e in edges.tolist()}

    assert observed == expected
    assert np.allclose(vertices, node_centers)


def test_by_wavefront_keep_loops_removed_kwargs_raise_typeerror():
    mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=8)
    with pytest.raises(TypeError):
        wave.by_wavefront_keep_loops(mesh, waves=1, progress=False, return_swc=False)

    with pytest.raises(TypeError):
        wave.by_wavefront_keep_loops(mesh, waves=1, progress=False, post_trim_rounds=1)


def test_by_wavefront_keep_loops_invalid_radius_agg_raises():
    mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=8)
    with pytest.raises(AssertionError, match=r'Unknown `radius_agg`'):
        wave.by_wavefront_keep_loops(mesh, waves=1, progress=False, radius_agg='nope')
