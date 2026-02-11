import numpy as np

from skeletor.utilities import extract_loop_node_sets


def test_extract_loop_node_sets_triangle():
    # Loopy graph: triangle 0-1-2-0
    loop_edges = np.array([[0, 1], [1, 2], [0, 2]], dtype=int)
    loop_node_centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    tree_edges = np.array([[0, 1], [1, 2]], dtype=int)

    out = extract_loop_node_sets(loop_edges=loop_edges,
                                 loop_node_centers=loop_node_centers,
                                 tree_edges_loop_index=tree_edges)

    assert out['counts']['n_loops'] == 1
    assert out['counts']['n_unique_nodes_in_loops'] == 3
    assert set(out['loops'][0]['node_ids'].tolist()) == {0, 1, 2}


def test_extract_loop_node_sets_square_with_diagonal():
    # Loopy graph edges: square + diagonal (0, 2)
    loop_edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0], [0, 2]
    ], dtype=int)
    loop_node_centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    # Tree excludes diagonal, so dropped edge is (0, 2)
    tree_edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=int)

    out = extract_loop_node_sets(loop_edges=loop_edges,
                                 loop_node_centers=loop_node_centers,
                                 tree_edges_loop_index=tree_edges)

    assert out['counts']['n_loops'] == 2

    loops = [set(item['node_ids'].tolist()) for item in out['loops']]
    assert {0, 1, 2} in loops
    assert {0, 1, 2, 3} in loops


def test_extract_loop_node_sets_empty_dropped_edges():
    loop_edges = np.array([[0, 1], [1, 2]], dtype=int)
    loop_node_centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    tree_edges = loop_edges.copy()

    out = extract_loop_node_sets(loop_edges=loop_edges,
                                 loop_node_centers=loop_node_centers,
                                 tree_edges_loop_index=tree_edges)

    assert out['loops'] == []
    assert out['counts']['n_loops'] == 0
    assert out['counts']['n_unique_nodes_in_loops'] == 0
