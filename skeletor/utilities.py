#    This script is part of skeletor (http://www.github.com/navis-org/skeletor).
#    Copyright (C) 2018 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.

import trimesh as tm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .skeletonize.base import Skeleton


def make_trimesh(mesh, validate=True, **kwargs):
    """Construct ``trimesh.Trimesh`` from input data.

    Parameters
    ----------
    meshdata :      tuple | dict | mesh-like object
                    Tuple: (vertices, faces)
                    dict: {'vertices': [], 'faces': []}
                    mesh-like object: mesh.vertices, mesh.faces
    validate :      bool
                    If True, will try to fix potential issues with the mesh
                    (e.g. infinite values, duplicate vertices, degenerate faces).
    **kwargs
                    Keyword arguments are passed through to
                    `skeletor.pre.fix_mesh` if `validate=True`.

    Returns
    -------
    trimesh.Trimesh

    """
    from .pre import fix_mesh

    if isinstance(mesh, tm.Trimesh):
        pass
    elif isinstance(mesh, (tuple, list)):
        if len(mesh) == 2:
            mesh = tm.Trimesh(vertices=mesh[0],
                              faces=mesh[1],
                              process=validate)
    elif isinstance(mesh, dict):
        mesh = tm.Trimesh(vertices=mesh['vertices'],
                          faces=mesh['faces'],
                          process=validate)
    elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
        mesh = tm.Trimesh(vertices=mesh.vertices,
                          faces=mesh.faces,
                          process=validate)
    else:
        raise TypeError('Unable to construct a trimesh.Trimesh from object of '
                        f'type "{type(mesh)}"')

    if validate:
        mesh = fix_mesh(mesh, inplace=True, **kwargs)

    return mesh


def canonicalize_undirected_edges(edges: np.ndarray) -> set[tuple[int, int]]:
    """Return set of undirected edges as sorted ``(u, v)`` tuples."""
    if edges is None:
        return set()

    arr = np.asarray(edges, dtype=int)
    if arr.size == 0:
        return set()

    arr = arr.reshape(-1, 2)
    arr = np.sort(arr, axis=1)
    return {tuple(e) for e in arr.tolist()}


def swc_parent_edges(swc_df: pd.DataFrame) -> np.ndarray:
    """Extract parent-child edges from SWC rows with valid parent IDs.

    Notes
    -----
    SWC stores trees, so parent links cannot represent cycles directly.
    """
    required = {'node_id', 'parent_id'}
    missing = required.difference(swc_df.columns)
    if missing:
        raise ValueError(f'SWC dataframe missing required columns: {sorted(missing)}')

    valid = swc_df.parent_id.values >= 0
    if not np.any(valid):
        return np.empty((0, 2), dtype=int)

    return swc_df.loc[valid, ['node_id', 'parent_id']].to_numpy(dtype=int)


def align_swc_nodes_to_loop_nodes(loop_node_centers: np.ndarray,
                                  swc_xyz: np.ndarray) -> np.ndarray:
    """Map each SWC node position to nearest loop-node index via ``cKDTree``."""
    loop_node_centers = np.asarray(loop_node_centers, dtype=float)
    swc_xyz = np.asarray(swc_xyz, dtype=float)

    if loop_node_centers.ndim != 2 or loop_node_centers.shape[1] != 3:
        raise ValueError('`loop_node_centers` must be shape (N, 3)')
    if swc_xyz.ndim != 2 or swc_xyz.shape[1] != 3:
        raise ValueError('`swc_xyz` must be shape (M, 3)')

    tree = cKDTree(loop_node_centers)
    _, idx = tree.query(swc_xyz, k=1)
    return idx.astype(int)


def compute_loop_edge_diff(loop_edges: np.ndarray,
                           loop_node_centers: np.ndarray,
                           tree_skeleton: 'Skeleton | None' = None,
                           tree_swc: pd.DataFrame | None = None,
                           tree_edges_loop_index: np.ndarray | None = None) -> dict:
    """Compute loop edges removed to obtain a tree (SWC cannot encode cycles)."""
    loop_edges = np.asarray(loop_edges, dtype=int).reshape(-1, 2)
    n_nodes = int(np.asarray(loop_node_centers).shape[0])

    provided = [tree_edges_loop_index is not None, tree_skeleton is not None, tree_swc is not None]
    if sum(provided) == 0:
        raise ValueError('Provide one tree source: tree_edges_loop_index, tree_skeleton, or tree_swc')

    if tree_edges_loop_index is not None:
        tree_edges = np.asarray(tree_edges_loop_index, dtype=int).reshape(-1, 2)
    else:
        swc = tree_swc if tree_swc is not None else tree_skeleton.swc
        swc_edges = swc_parent_edges(swc)

        if swc_edges.size == 0:
            tree_edges = np.empty((0, 2), dtype=int)
        else:
            if tree_skeleton is not None and hasattr(tree_skeleton, 'loop_tree_edges'):
                tree_edges = np.asarray(tree_skeleton.loop_tree_edges, dtype=int).reshape(-1, 2)
            else:
                swc_xyz = swc[['x', 'y', 'z']].to_numpy(dtype=float)
                swc_to_loop = align_swc_nodes_to_loop_nodes(loop_node_centers, swc_xyz)
                tree_edges = np.column_stack((
                    swc_to_loop[swc_edges[:, 0]],
                    swc_to_loop[swc_edges[:, 1]],
                )).astype(int)

    loop_set = canonicalize_undirected_edges(loop_edges)
    tree_set = canonicalize_undirected_edges(tree_edges)
    dropped_set = loop_set - tree_set

    dropped_edges = (np.array(sorted(dropped_set), dtype=int)
                     if dropped_set else np.empty((0, 2), dtype=int))
    tree_edges_unique = (np.array(sorted(tree_set), dtype=int)
                         if tree_set else np.empty((0, 2), dtype=int))
    loop_edges_all = (np.array(sorted(loop_set), dtype=int)
                      if loop_set else np.empty((0, 2), dtype=int))

    return {
        'tree_edges': tree_edges_unique,
        'dropped_edges': dropped_edges,
        'loop_edges_all': loop_edges_all,
        'counts': {
            'N': n_nodes,
            'E_all': int(loop_edges_all.shape[0]),
            'E_tree': int(tree_edges_unique.shape[0]),
            'E_dropped': int(dropped_edges.shape[0]),
        },
    }


def _tree_path_nodes(tree_edges: np.ndarray,
                     source: int,
                     target: int) -> list[int]:
    """Find path nodes between ``source`` and ``target`` in an undirected tree."""
    if source == target:
        return [int(source)]

    tree_edges = np.asarray(tree_edges, dtype=int).reshape(-1, 2)
    adjacency: dict[int, set[int]] = {}
    for u, v in tree_edges:
        adjacency.setdefault(int(u), set()).add(int(v))
        adjacency.setdefault(int(v), set()).add(int(u))

    if source not in adjacency or target not in adjacency:
        return []

    queue = [int(source)]
    parent = {int(source): -1}
    i = 0

    while i < len(queue):
        node = queue[i]
        i += 1
        if node == target:
            break
        for nb in adjacency.get(node, ()):
            if nb in parent:
                continue
            parent[nb] = node
            queue.append(nb)

    if int(target) not in parent:
        return []

    path = [int(target)]
    while path[-1] != int(source):
        path.append(parent[path[-1]])
    path.reverse()
    return path


def extract_loop_node_sets(loop_edges: np.ndarray,
                           loop_node_centers: np.ndarray,
                           tree_skeleton: 'Skeleton | None' = None,
                           tree_swc: pd.DataFrame | None = None,
                           tree_edges_loop_index: np.ndarray | None = None,
                           deduplicate: bool = True) -> dict:
    """Return node IDs involved in each loop/cycle implied by dropped edges.

    Notes
    -----
    SWC is tree-based and cannot encode cycles directly. Loops are therefore
    inferred as *fundamental cycles*: for each edge dropped when converting the
    loopy graph to a tree, we recover the unique path between that edge's
    endpoints in the tree and combine both.

    If ``tree_edges_loop_index`` is not provided and no
    ``tree_skeleton.loop_tree_edges`` is available, SWC nodes are aligned to
    loop nodes using nearest-neighbour mapping. In that case, ambiguous matches
    can merge nearby nodes and affect inferred loops.

    Examples
    --------
    A triangle with tree edges ``(0, 1)`` and ``(1, 2)`` has dropped edge
    ``(0, 2)``. The inferred loop node IDs are ``[0, 1, 2]``.
    """
    diff = compute_loop_edge_diff(loop_edges=loop_edges,
                                  loop_node_centers=loop_node_centers,
                                  tree_skeleton=tree_skeleton,
                                  tree_swc=tree_swc,
                                  tree_edges_loop_index=tree_edges_loop_index)

    tree_edges = diff['tree_edges']
    dropped_edges = diff['dropped_edges']
    n_nodes = int(np.asarray(loop_node_centers).shape[0])

    loops = []
    seen: set[tuple[int, ...]] = set()

    for edge in dropped_edges:
        u, v = (int(edge[0]), int(edge[1]))
        path_nodes = _tree_path_nodes(tree_edges, source=u, target=v)

        if path_nodes:
            node_ids = np.array(sorted(set(path_nodes)), dtype=int)
            path_edges = np.array(list(zip(path_nodes[:-1], path_nodes[1:])), dtype=int)
        else:
            node_ids = np.array(sorted({u, v}), dtype=int)
            path_edges = np.empty((0, 2), dtype=int)

        key = tuple(node_ids.tolist())
        if deduplicate and key in seen:
            continue
        seen.add(key)

        loops.append({
            'loop_id': len(loops),
            'dropped_edge': (u, v),
            'node_ids': node_ids,
            'path_edges': path_edges,
        })

    unique_nodes = (np.unique(np.concatenate([l['node_ids'] for l in loops]))
                    if loops else np.array([], dtype=int))

    return {
        'loops': loops,
        'counts': {
            'n_loops': int(len(loops)),
            'n_nodes_total': n_nodes,
            'n_unique_nodes_in_loops': int(unique_nodes.size),
        }
    }


def loop_node_sets_from_result(loop_result: dict | None = None,
                               skeleton: 'Skeleton | None' = None,
                               deduplicate: bool = True) -> dict:
    """Convenience wrapper for inferring cycle node sets from wavefront output.

    This mirrors ``visualize_added_loops`` input patterns and returns
    ``extract_loop_node_sets`` output.
    """
    if skeleton is None and loop_result is None:
        raise ValueError('Provide `skeleton` and/or `loop_result`.')

    if skeleton is not None and hasattr(skeleton, 'loop_edges') and hasattr(skeleton, 'loop_node_centers'):
        loop_edges = np.asarray(skeleton.loop_edges, dtype=int)
        loop_node_centers = np.asarray(skeleton.loop_node_centers, dtype=float)
        loop_tree_edges = getattr(skeleton, 'loop_tree_edges', None)
        return extract_loop_node_sets(loop_edges=loop_edges,
                                      loop_node_centers=loop_node_centers,
                                      tree_skeleton=skeleton,
                                      tree_edges_loop_index=loop_tree_edges,
                                      deduplicate=deduplicate)

    if loop_result is None or skeleton is None:
        raise ValueError('When skeleton has no loop metadata, provide both `loop_result` and `skeleton`.')

    loop_edges = np.asarray(loop_result['edges'], dtype=int)
    loop_node_centers = np.asarray(loop_result['node_centers'], dtype=float)
    return extract_loop_node_sets(loop_edges=loop_edges,
                                  loop_node_centers=loop_node_centers,
                                  tree_skeleton=skeleton,
                                  deduplicate=deduplicate)


def derive_loop_to_swc_node_map(skeleton: 'Skeleton') -> dict[int, int]:
    """Derive loop-node ID -> SWC node ID mapping via mesh correspondences.

    Notes
    -----
    Loop-node IDs live in loop-graph index space (``loop_node_centers`` /
    ``loop_edges``) and are not guaranteed to match SWC ``node_id`` values,
    because SWC can be reindexed during tree construction.

    Mapping is derived explicitly using:
      * ``loop_vertex_to_node_map``: mesh vertex -> loop-node ID
      * ``mesh_map``: mesh vertex -> SWC node ID

    If multiple mesh vertices vote for different SWC IDs for the same loop node,
    the most frequent SWC ID is chosen (ties broken by smaller ID).
    """
    if not hasattr(skeleton, 'loop_vertex_to_node_map'):
        raise ValueError('Skeleton is missing `loop_vertex_to_node_map`.')
    if getattr(skeleton, 'mesh_map', None) is None:
        raise ValueError('Skeleton is missing `mesh_map`.')

    loop_vertex_to_node_map = np.asarray(skeleton.loop_vertex_to_node_map, dtype=int).reshape(-1)
    mesh_map = np.asarray(skeleton.mesh_map, dtype=int).reshape(-1)

    if loop_vertex_to_node_map.shape[0] != mesh_map.shape[0]:
        raise ValueError('`loop_vertex_to_node_map` and `mesh_map` must have matching lengths.')

    mapping: dict[int, int] = {}
    for loop_id in np.unique(loop_vertex_to_node_map):
        swc_ids = mesh_map[loop_vertex_to_node_map == loop_id]
        if swc_ids.size == 0:
            continue
        values, counts = np.unique(swc_ids, return_counts=True)
        best = values[np.flatnonzero(counts == counts.max())].min()
        mapping[int(loop_id)] = int(best)

    return mapping


def _orient_edges_to_parent_map(edges_local: np.ndarray,
                                n_nodes: int,
                                root_local: int) -> np.ndarray:
    """Build deterministic SWC parent array from local undirected edges."""
    parents = np.full(n_nodes, -1, dtype=int)
    if n_nodes == 0:
        return parents

    adjacency: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for edge in np.asarray(edges_local, dtype=int).reshape(-1, 2):
        u, v = int(edge[0]), int(edge[1])
        if u < 0 or v < 0 or u >= n_nodes or v >= n_nodes or u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    seen = {int(root_local)}
    queue = [int(root_local)]
    i = 0

    while i < len(queue):
        node = queue[i]
        i += 1
        for nb in sorted(adjacency.get(node, ())):
            if nb in seen:
                continue
            seen.add(nb)
            parents[nb] = node + 1
            queue.append(nb)

    for node in range(n_nodes):
        if node in seen:
            continue
        seen.add(node)
        queue = [node]
        j = 0
        while j < len(queue):
            cur = queue[j]
            j += 1
            for nb in sorted(adjacency.get(cur, ())):
                if nb in seen:
                    continue
                seen.add(nb)
                parents[nb] = cur + 1
                queue.append(nb)

    return parents


def export_loops_to_swc(loop_info: dict,
                        loop_node_centers: np.ndarray,
                        loop_node_radii: np.ndarray | None = None,
                        out_dir: str = 'loop_swcs',
                        file_prefix: str = 'loop',
                        break_on_dropped_edge: bool = True,
                        dropped_edge_name_in_header: bool = True) -> list[str]:
    """Export one SWC per inferred loop from ``loop_info['loops']``."""
    loops = list(loop_info.get('loops', []))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    centers = np.asarray(loop_node_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError('`loop_node_centers` must be shape (N, 3).')

    radii = None
    if loop_node_radii is not None:
        radii = np.asarray(loop_node_radii, dtype=float).reshape(-1)
        if radii.shape[0] != centers.shape[0]:
            raise ValueError('`loop_node_radii` length must match loop nodes.')

    if not loops:
        return []

    written: list[str] = []
    for loop in loops:
        loop_id = int(loop['loop_id'])
        node_ids = np.asarray(loop.get('node_ids', []), dtype=int).reshape(-1)
        path_edges = np.asarray(loop.get('path_edges', []), dtype=int).reshape(-1, 2)
        dropped_edge = tuple(np.asarray(loop.get('dropped_edge', (-1, -1)), dtype=int).tolist())

        if node_ids.size == 0:
            continue
        if np.any(node_ids < 0) or np.any(node_ids >= centers.shape[0]):
            raise ValueError(
                'Loop `node_ids` are out of bounds for `loop_node_centers`; '
                'use loop-graph IDs, not SWC node IDs.'
            )

        node_ids = np.array(sorted(set(node_ids.tolist())), dtype=int)
        global_to_local = {nid: i for i, nid in enumerate(node_ids.tolist())}

        local_edges = []
        for u, v in path_edges:
            u_i = global_to_local.get(int(u))
            v_i = global_to_local.get(int(v))
            if u_i is None or v_i is None:
                continue
            local_edges.append((u_i, v_i))

        if not break_on_dropped_edge and len(dropped_edge) == 2:
            du_i = global_to_local.get(int(dropped_edge[0]))
            dv_i = global_to_local.get(int(dropped_edge[1]))
            if du_i is not None and dv_i is not None and du_i != dv_i:
                local_edges.append((du_i, dv_i))

        if dropped_edge and int(dropped_edge[0]) in global_to_local:
            root_local = global_to_local[int(dropped_edge[0])]
        else:
            root_local = 0

        parents = _orient_edges_to_parent_map(np.asarray(local_edges, dtype=int),
                                              n_nodes=node_ids.size,
                                              root_local=root_local)

        local_centers = centers[node_ids]
        local_radii = radii[node_ids] if radii is not None else np.zeros(node_ids.size, dtype=float)

        file_path = out_path / f'{file_prefix}_{loop_id:03d}.swc'
        with file_path.open('w') as f:
            f.write('# SWC format file\n')
            f.write('# Exported from loop node sets (loop-graph ID space)\n')
            if dropped_edge_name_in_header and len(dropped_edge) == 2:
                f.write(f'# dropped_edge: {int(dropped_edge[0])} {int(dropped_edge[1])}\n')
            for i in range(node_ids.size):
                x, y, z = local_centers[i]
                row = [i + 1, 0, x, y, z, local_radii[i], int(parents[i])]
                f.write(' '.join(map(str, row)) + '\n')

        written.append(str(file_path))

    return written


def export_loops_to_swc_from_skeleton(skeleton: 'Skeleton',
                                      loop_result: dict | None = None,
                                      out_dir: str = 'loop_swcs',
                                      file_prefix: str = 'loop',
                                      deduplicate: bool = True,
                                      break_on_dropped_edge: bool = True) -> list[str]:
    """Infer loop node sets and export one SWC per loop.

    Loop node IDs from loop utilities are interpreted in loop-graph index
    space and exported from ``skeleton.loop_node_centers`` / radii where
    available.
    """
    if skeleton is None:
        raise ValueError('Provide `skeleton`.')

    loop_info = loop_node_sets_from_result(loop_result=loop_result,
                                           skeleton=skeleton,
                                           deduplicate=deduplicate)

    if hasattr(skeleton, 'loop_node_centers'):
        loop_node_centers = np.asarray(skeleton.loop_node_centers, dtype=float)
        loop_node_radii = getattr(skeleton, 'loop_node_radii', None)
    else:
        if loop_result is None:
            raise ValueError('When skeleton has no loop metadata, provide `loop_result`.')
        loop_node_centers = np.asarray(loop_result['node_centers'], dtype=float)
        loop_node_radii = loop_result.get('node_radii', None)

    return export_loops_to_swc(loop_info=loop_info,
                               loop_node_centers=loop_node_centers,
                               loop_node_radii=loop_node_radii,
                               out_dir=out_dir,
                               file_prefix=file_prefix,
                               break_on_dropped_edge=break_on_dropped_edge)


def plot_loop_vs_tree_3d(loop_node_centers: np.ndarray,
                         tree_edges: np.ndarray,
                         dropped_edges: np.ndarray,
                         node_sample: int = 20000,
                         tree_edge_sample: int = 20000,
                         dropped_edge_sample: int = 20000,
                         show_nodes: bool = True) -> None:
    """Plot loop-vs-tree edge diff in 3D using matplotlib only."""
    pts = np.asarray(loop_node_centers, dtype=float)
    tree_edges = np.asarray(tree_edges, dtype=int).reshape(-1, 2)
    dropped_edges = np.asarray(dropped_edges, dtype=int).reshape(-1, 2)

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1985)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    if show_nodes and pts.shape[0] > 0:
        n = min(node_sample, pts.shape[0])
        idx = rng.choice(pts.shape[0], size=n, replace=False)
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], s=1, alpha=0.08, color='k')

    if tree_edges.shape[0] > 0:
        e = min(tree_edge_sample, tree_edges.shape[0])
        eidx = rng.choice(tree_edges.shape[0], size=e, replace=False)
        for u, v in tree_edges[eidx]:
            seg = pts[[u, v]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color='tab:blue', alpha=0.2, linewidth=0.6)

    if dropped_edges.shape[0] > 0:
        e = min(dropped_edge_sample, dropped_edges.shape[0])
        eidx = rng.choice(dropped_edges.shape[0], size=e, replace=False)
        for u, v in dropped_edges[eidx]:
            seg = pts[[u, v]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color='tab:red', alpha=0.8, linewidth=1.4)

    ax.set_title('Wavefront contracted graph: tree edges vs dropped loop edges')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()


def visualize_added_loops(loop_result: dict | None = None,
                          skeleton: 'Skeleton | None' = None) -> dict:
    """Visualize edges dropped when converting a loopy graph into SWC tree.

    SWC is tree-based and cannot encode cycles, so loop edges are removed during
    tree extraction. This helper computes and plots those dropped edges.
    """
    if skeleton is None and loop_result is None:
        raise ValueError('Provide `skeleton` and/or `loop_result`.')

    if skeleton is not None and hasattr(skeleton, 'loop_edges') and hasattr(skeleton, 'loop_node_centers'):
        loop_edges = np.asarray(skeleton.loop_edges, dtype=int)
        loop_node_centers = np.asarray(skeleton.loop_node_centers, dtype=float)
        loop_tree_edges = getattr(skeleton, 'loop_tree_edges', None)
        diff = compute_loop_edge_diff(loop_edges=loop_edges,
                                      loop_node_centers=loop_node_centers,
                                      tree_skeleton=skeleton,
                                      tree_edges_loop_index=loop_tree_edges)
    else:
        if loop_result is None or skeleton is None:
            raise ValueError('When skeleton has no loop metadata, provide both `loop_result` and `skeleton`.')

        loop_edges = np.asarray(loop_result['edges'], dtype=int)
        loop_node_centers = np.asarray(loop_result['node_centers'], dtype=float)
        diff = compute_loop_edge_diff(loop_edges=loop_edges,
                                      loop_node_centers=loop_node_centers,
                                      tree_skeleton=skeleton)

    plot_loop_vs_tree_3d(loop_node_centers=loop_node_centers,
                         tree_edges=diff['tree_edges'],
                         dropped_edges=diff['dropped_edges'])
    return diff
