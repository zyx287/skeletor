from pathlib import Path

import numpy as np
import pandas as pd

from skeletor.skeletonize.base import Skeleton
from skeletor.utilities import (
    derive_loop_to_swc_node_map,
    export_loops_to_swc,
    export_loops_to_swc_from_skeleton,
)


def _read_swc_data_rows(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        rows.append(line.split())
    return rows


def test_export_loops_to_swc_single_loop(tmp_path):
    loop_info = {
        'loops': [
            {
                'loop_id': 0,
                'dropped_edge': (0, 2),
                'node_ids': np.array([0, 1, 2], dtype=int),
                'path_edges': np.array([[0, 1], [1, 2]], dtype=int),
            }
        ],
        'counts': {'n_loops': 1},
    }
    centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    radii = np.array([0.1, 0.2, 0.3])

    paths = export_loops_to_swc(loop_info=loop_info,
                                loop_node_centers=centers,
                                loop_node_radii=radii,
                                out_dir=str(tmp_path),
                                file_prefix='loop')

    assert len(paths) == 1
    out_file = Path(paths[0])
    assert out_file.name == 'loop_000.swc'

    text = out_file.read_text()
    assert '# dropped_edge: 0 2' in text

    rows = _read_swc_data_rows(out_file)
    assert len(rows) == 3
    assert rows[0][-1] == '-1'
    assert rows[1][-1] == '1'
    assert rows[2][-1] == '2'


def test_export_loops_to_swc_multiple_loops(tmp_path):
    loop_info = {
        'loops': [
            {
                'loop_id': 0,
                'dropped_edge': (0, 2),
                'node_ids': np.array([0, 1, 2], dtype=int),
                'path_edges': np.array([[0, 1], [1, 2]], dtype=int),
            },
            {
                'loop_id': 1,
                'dropped_edge': (2, 4),
                'node_ids': np.array([2, 3, 4], dtype=int),
                'path_edges': np.array([[2, 3], [3, 4]], dtype=int),
            },
        ],
        'counts': {'n_loops': 2},
    }
    centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])

    paths = export_loops_to_swc(loop_info=loop_info,
                                loop_node_centers=centers,
                                out_dir=str(tmp_path),
                                file_prefix='lp')

    assert len(paths) == 2
    assert {Path(p).name for p in paths} == {'lp_000.swc', 'lp_001.swc'}


def test_export_loops_to_swc_empty_loops(tmp_path):
    out = export_loops_to_swc(loop_info={'loops': [], 'counts': {'n_loops': 0}},
                              loop_node_centers=np.zeros((0, 3), dtype=float),
                              out_dir=str(tmp_path))
    assert out == []
    assert tmp_path.exists()


def test_loop_to_swc_map_and_export_uses_loop_space_ids(tmp_path):
    swc = pd.DataFrame([
        {'node_id': 100, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'parent_id': -1},
        {'node_id': 200, 'x': 1.0, 'y': 0.0, 'z': 0.0, 'parent_id': 100},
    ])
    skeleton = Skeleton(swc=swc, mesh_map=np.array([100, 100, 200, 200], dtype=int))
    skeleton.loop_vertex_to_node_map = np.array([0, 0, 1, 1], dtype=int)
    skeleton.loop_node_centers = np.array([
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
    ])
    skeleton.loop_node_radii = np.array([0.5, 0.6])
    skeleton.loop_edges = np.array([[0, 1]], dtype=int)
    skeleton.loop_tree_edges = np.array([], dtype=int).reshape(0, 2)

    mapping = derive_loop_to_swc_node_map(skeleton)
    assert mapping == {0: 100, 1: 200}
    assert 0 not in set(swc.node_id.values)

    loop_result = {
        'edges': np.array([[0, 1]], dtype=int),
        'node_centers': skeleton.loop_node_centers,
    }
    paths = export_loops_to_swc_from_skeleton(skeleton=skeleton,
                                              loop_result=loop_result,
                                              out_dir=str(tmp_path),
                                              file_prefix='mapped')

    assert len(paths) == 1

    auto_rows = _read_swc_data_rows(Path(paths[0]))
    assert len(auto_rows) == 2

    loop_info = {
        'loops': [{
            'loop_id': 0,
            'dropped_edge': (0, 1),
            'node_ids': np.array([0, 1], dtype=int),
            'path_edges': np.array([[0, 1]], dtype=int),
        }],
        'counts': {'n_loops': 1},
    }
    direct_paths = export_loops_to_swc(loop_info=loop_info,
                                       loop_node_centers=skeleton.loop_node_centers,
                                       loop_node_radii=skeleton.loop_node_radii,
                                       out_dir=str(tmp_path),
                                       file_prefix='mapped')
    assert len(direct_paths) == 1

    rows = _read_swc_data_rows(Path(direct_paths[0]))
    assert len(rows) == 2
    assert rows[0][-1] == '-1'
    assert rows[1][-1] == '1'
