import numpy as np
import pandas as pd
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


def test_by_wavefront_keep_loops_defaults_do_not_run_post(monkeypatch):
    mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=8)

    monkeypatch.setattr(wave, 'trim_terminal_nodes',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('trim should not run')))
    monkeypatch.setattr(wave, 'collapse_soma_nodes',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('collapse should not run')))

    out = wave.by_wavefront_keep_loops(mesh, waves=1, progress=False, return_swc=False)
    assert {'node_centers', 'node_radii', 'edges', 'mesh_map'} <= set(out)


def test_by_wavefront_keep_loops_post_sequence(monkeypatch):
    mesh = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=8)
    calls = []

    def _trim(obj, rounds=1, **kwargs):
        calls.append(('trim', rounds))
        return obj

    def _collapse(obj, soma_mesh=None, **kwargs):
        calls.append(('collapse', soma_mesh))
        return obj

    monkeypatch.setattr(wave, 'trim_terminal_nodes', _trim)
    monkeypatch.setattr(wave, 'collapse_soma_nodes', _collapse)

    wave.by_wavefront_keep_loops(mesh,
                                 waves=1,
                                 progress=False,
                                 return_swc=False,
                                 post_trim_rounds=2,
                                 post_collapse_soma=True,
                                 post_soma_mesh=object())

    assert calls[0][0] == 'trim'
    assert calls[1][0] == 'collapse'
    assert calls[2][0] == 'trim'
