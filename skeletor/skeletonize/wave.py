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

import igraph as ig
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from tqdm.auto import tqdm

from ..utilities import make_trimesh
from .base import Skeleton
from .utils import make_swc, reindex_swc, edges_to_graph

__all__ = ['by_wavefront', 'by_wavefront_keep_loops', 'select_wave_origins_from_soma',
          'compute_robust_dist_to_soma', 'repair_tree_local_swaps']

# This flag determines whether we use inverse radii or edge lengths for the MST.
# Radii make more sense if working with tubular structures like neurons but this
# might not apply for all scenarios
PRESERVE_BACKBONE = True


def by_wavefront(mesh,
                 waves=1,
                 origins=None,
                 step_size=1,
                 radius_agg='mean',
                 progress=True,
                 soma_mesh=None,
                 origin_from_soma: bool = False,
                 origin_strategy: str = 'soma_surface_fps',
                 loop_break: str = 'auto',
                 loop_params: dict = None):
    """Skeletonize a mesh using wave fronts.

    The algorithm tries to find rings of vertices and collapse them to
    their center. This is done by propagating a wave across the mesh starting at
    a single seed vertex. As the wave travels across the mesh we keep track of
    which vertices are are encountered at each step. Groups of connected
    vertices that are "hit" by the wave at the same time are considered rings
    and subsequently collapsed. By its nature this works best with tubular meshes.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to be skeletonize. Can an object that has
                    ``.vertices`` and ``.faces`` properties  (e.g. a
                    trimesh.Trimesh) or a tuple ``(vertices, faces)`` or a
                    dictionary ``{'vertices': vertices, 'faces': faces}``.
    waves :         int
                    Number of waves to run across the mesh. Each wave is
                    initialized at a different vertex which produces slightly
                    different rings. The final skeleton is produced from a mean
                    across all waves. More waves produce higher resolution
                    skeletons but also introduce more noise.
    origins :       int | list of ints, optional
                    Vertex ID(s) where the wave(s) are initialized. If we run
                    out of origins (either because less `origins` than `waves`
                    or because no origin for one of the connected components)
                    will fall back to semi-random origin.
    step_size :     int
                    Values greater 1 effectively lead to binning of rings. For
                    example a stepsize of 2 means that two adjacent vertex rings
                    will be collapsed to the same center. This can help reduce
                    noise in the skeleton (and as such counteracts a large
                    number of waves).
    radius_agg :    "mean" | "median" | "max" | "min" | "percentile75" | "percentile25"
                    Function used to aggregate radii over sample (i.e. the
                    vertices forming a ring that we collapse to its center).
    progress :      bool
                    If True, will show progress bar.
    soma_mesh :     mesh obj, optional
                    Soma mesh used for soma-aware origin selection and soma-aware
                    loop breaking.
    origin_from_soma : bool
                    If True and ``origins`` is None, wave origins are selected
                    from ``soma_mesh`` via ``select_wave_origins_from_soma``.
    origin_strategy : str
                    Origin selection strategy passed to
                    ``select_wave_origins_from_soma``.
    loop_break :    "auto" | "mst" | "soma_farthest" | "soma_rooted"
                    Cycle-breaking strategy after graph contraction. ``auto``
                    uses ``soma_rooted`` when soma information is provided,
                    otherwise falls back to historical ``mst`` behavior.
    loop_params :   dict, optional
                    Parameters for soma-aware loop breaking.

    Returns
    -------
    skeletor.Skeleton
                    Holds results of the skeletonization and enables quick
                    visualization.

    """
    agg_map = {'mean': np.mean, 'max': np.max, 'min': np.min,
               'median': np.median,
               'percentile75': lambda x: np.percentile(x, 75),
               'percentile25': lambda x: np.percentile(x, 25)}
    assert radius_agg in agg_map, f'Unknown `radius_agg`: "{radius_agg}"'
    rad_agg_func = agg_map[radius_agg]

    if loop_break not in {'auto', 'mst', 'soma_farthest', 'soma_rooted'}:
        raise ValueError(f'Unknown `loop_break`: "{loop_break}"')

    loop_params = {} if loop_params is None else dict(loop_params)
    if loop_break == 'auto':
        loop_break = 'soma_rooted' if soma_mesh is not None else 'mst'

    if (origin_from_soma or loop_break in {'soma_farthest', 'soma_rooted'}) and soma_mesh is None:
        raise ValueError('`soma_mesh` is required for soma-aware origins and loop breaking')

    mesh = make_trimesh(mesh, validate=False)

    if origin_from_soma and origins is None:
        origins = select_wave_origins_from_soma(cell_mesh=mesh,
                                                soma_mesh=soma_mesh,
                                                waves=waves,
                                                strategy=origin_strategy,
                                                seed=int(loop_params.get('seed', 1985)),
                                                k_candidates=int(loop_params.get('k_candidates', 5000)))

    node_centers, node_radii, G, vertex_to_node_map = _wavefront_contracted_graph(
        mesh=mesh,
        waves=waves,
        origins=origins,
        step_size=step_size,
        rad_agg_func=rad_agg_func,
        progress=progress,
        strict_origins=origin_from_soma and origins is not None,
    )

    tree_edges = _wavefront_tree_edges(
        G=G,
        node_centers=node_centers,
        node_radii=node_radii,
        loop_break=loop_break,
        soma_mesh=soma_mesh,
        loop_params=loop_params,
    )

    # Create a directed acyclic and hierarchical graph
    G_nx = edges_to_graph(edges=np.array(tree_edges, dtype=int),
                          nodes=np.arange(0, len(G.vs)),
                          fix_tree=True,  # this makes sure graph is oriented
                          drop_disconnected=False)

    # Generate the SWC table
    swc = make_swc(G_nx, coords=node_centers, reindex=False, validate=True)
    swc['radius'] = node_radii[swc.node_id.values]
    _, new_ids = reindex_swc(swc, inplace=True)

    # Update vertex to node ID map
    vertex_to_node_map = np.array([new_ids[n] for n in vertex_to_node_map])

    return Skeleton(swc=swc, mesh=mesh, mesh_map=vertex_to_node_map,
                    method='wavefront')


def by_wavefront_keep_loops(mesh,
                            waves=1,
                            origins=None,
                            step_size=1,
                            radius_agg='mean',
                            progress=True):
    """Skeletonize a mesh by wavefront contraction while preserving loops.

    Unlike :func:`by_wavefront`, this function does not extract a tree or
    convert results to SWC. It returns the contracted loop graph directly as
    ``(edges, vertices)`` in the same order as
    :meth:`skeletor.skeletonize.base.Skeleton.mend_breaks`.

    Parameters
    ----------
    mesh :          mesh obj
                    The mesh to skeletonize.
    waves :         int
                    Number of wavefront propagations.
    origins :       int | list of ints, optional
                    Seed vertex ID(s) for waves.
    step_size :     int
                    Ring binning size.
    radius_agg :    "mean" | "median" | "max" | "min" | "percentile75" | "percentile25"
                    Function used to aggregate ring radii during contraction.
    progress :      bool
                    If True, shows progress bar.

    Returns
    -------
    edges :         (N, 2) array
                    Edges of the contracted loop-preserving graph.
    vertices :      (M, 3) array
                    Coordinates of contracted graph vertices.
    """
    agg_map = {'mean': np.mean, 'max': np.max, 'min': np.min,
               'median': np.median,
               'percentile75': lambda x: np.percentile(x, 75),
               'percentile25': lambda x: np.percentile(x, 25)}
    assert radius_agg in agg_map, f'Unknown `radius_agg`: "{radius_agg}"'
    rad_agg_func = agg_map[radius_agg]

    mesh = make_trimesh(mesh, validate=False)

    node_centers, _, G, _ = _wavefront_contracted_graph(
        mesh=mesh,
        waves=waves,
        origins=origins,
        step_size=step_size,
        rad_agg_func=rad_agg_func,
        progress=progress,
        strict_origins=False,
    )

    loop_edges = np.asarray(G.get_edgelist(), dtype=int).reshape(-1, 2)
    vertices = np.asarray(node_centers, dtype=float)
    return loop_edges, vertices


def _wavefront_contracted_graph(mesh, waves, origins, step_size, rad_agg_func,
                                progress, strict_origins):
    """Run wavefront casting and return the simplified contracted graph."""
    centers_final, radii_final, G = _cast_waves(mesh, waves=waves,
                                                origins=origins,
                                                step_size=step_size,
                                                rad_agg_func=rad_agg_func,
                                                progress=progress,
                                                strict_origins=strict_origins)

    node_centers, vertex_to_node_map = np.unique(centers_final,
                                                 return_inverse=True, axis=0)

    node_radii = pd.DataFrame()
    node_radii['node_id'] = vertex_to_node_map
    node_radii['radius'] = radii_final
    node_radii = node_radii.groupby('node_id').radius.apply(rad_agg_func).values

    G.contract_vertices(vertex_to_node_map)
    G = G.simplify()

    return node_centers, node_radii, G, vertex_to_node_map


def _wavefront_tree_edges(G, node_centers, node_radii, loop_break, soma_mesh, loop_params):
    """Extract a tree from the contracted graph using historical logic."""
    el = np.array(G.get_edgelist(), dtype=int)

    if len(el) == 0:
        return el

    if loop_break == 'mst':
        if PRESERVE_BACKBONE:
            weights = _edge_strength_weights(el=el,
                                             node_radii=node_radii,
                                             node_centers=node_centers)
        else:
            weights = np.linalg.norm(node_centers[el[:, 0]] - node_centers[el[:, 1]], axis=1)

        tree = G.spanning_tree(weights=1 / weights)
        return np.array(tree.get_edgelist(), dtype=int)

    soma_mesh = make_trimesh(soma_mesh, validate=False)
    return _soma_aware_tree_edges(G=G,
                                  node_centers=node_centers,
                                  node_radii=node_radii,
                                  soma_mesh=soma_mesh,
                                  mode=loop_break,
                                  loop_params=loop_params)


def select_wave_origins_from_soma(cell_mesh, soma_mesh, waves: int,
                                  strategy: str = 'soma_surface_fps',
                                  seed: int = 1985,
                                  k_candidates: int = 5000) -> np.ndarray:
    """Return soma-aware seed vertex IDs on the cell mesh."""
    cell_mesh = make_trimesh(cell_mesh, validate=False)
    soma_mesh = make_trimesh(soma_mesh, validate=False)

    waves = int(waves)
    if waves < 1:
        raise ValueError('`waves` must be integer >= 1')

    if strategy not in {'soma_surface_fps', 'soma_surface_random', 'soma_center_knn'}:
        raise ValueError(f'Unknown `strategy`: "{strategy}"')

    rng = np.random.RandomState(seed)
    cell_vertices = np.asarray(cell_mesh.vertices)
    soma_vertices = np.asarray(soma_mesh.vertices)

    if len(cell_vertices) == 0 or len(soma_vertices) == 0:
        return np.array([], dtype=int)

    tree = cKDTree(cell_vertices)
    soma_centroid = soma_vertices.mean(axis=0)

    if strategy in {'soma_surface_fps', 'soma_surface_random'}:
        nearest = tree.query(soma_vertices, k=1)[1]
        candidates = np.unique(np.asarray(nearest, dtype=int))
    else:
        k = min(len(cell_vertices), int(k_candidates))
        if k < 1:
            return np.array([], dtype=int)
        nearest = tree.query(soma_centroid.reshape(1, -1), k=k)[1]
        candidates = np.unique(np.asarray(nearest, dtype=int).reshape(-1))

    if len(candidates) == 0:
        return np.array([], dtype=int)

    n_select = min(waves, len(candidates))
    if strategy == 'soma_surface_random':
        return rng.choice(candidates, size=n_select, replace=False).astype(int)

    candidate_xyz = cell_vertices[candidates]
    start_ix = int(np.argmin(np.linalg.norm(candidate_xyz - soma_centroid, axis=1)))
    return _farthest_point_sample_indices(candidates,
                                          candidate_xyz,
                                          n_select=n_select,
                                          seed_pos=start_ix).astype(int)


def _edge_strength_weights(el, node_radii, node_centers):
    """Historical backbone-preserving weights for MST mode."""
    # Use the minimum radius between vertices in an edge
    weights_rad = np.vstack((node_radii[el[:, 0]],
                             node_radii[el[:, 1]])).mean(axis=0)

    # For each node generate a vector based on its immediate neighbors
    _, alpha = dotprops(node_centers)
    weights_alpha = np.vstack((alpha[el[:, 0]],
                               alpha[el[:, 1]])).mean(axis=0)

    # Combine both which means we are most likely to cut at small branches
    # outside of the backbone
    weights = weights_rad * weights_alpha

    # MST doesn't like 0 for weights
    if np.any(weights > 0):
        weights[weights <= 0] = weights[weights > 0].min() / 2
    else:
        weights[:] = 1

    return weights


def _farthest_point_sample_indices(candidates, candidate_xyz, n_select, seed_pos=0):
    """Farthest-point-sample candidate IDs in O(n_select * n_candidates)."""
    if n_select <= 0 or len(candidates) == 0:
        return np.array([], dtype=int)

    seed_pos = int(seed_pos)
    selected_pos = [seed_pos]
    min_dist = np.linalg.norm(candidate_xyz - candidate_xyz[seed_pos], axis=1)
    min_dist[seed_pos] = -np.inf

    while len(selected_pos) < n_select:
        nxt = int(np.argmax(min_dist))
        selected_pos.append(nxt)
        dist = np.linalg.norm(candidate_xyz - candidate_xyz[nxt], axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected_pos] = -np.inf

    return np.asarray(candidates)[np.asarray(selected_pos, dtype=int)]


def _soma_aware_tree_edges(G, node_centers, node_radii, soma_mesh, mode, loop_params):
    """Build tree edges with soma-aware cycle handling."""
    el = np.array(G.get_edgelist(), dtype=int)
    if len(el) == 0:
        return el

    eps = float(loop_params.get('eps', 1e-12))
    features = _compute_edge_features(el=el,
                                      node_centers=node_centers,
                                      node_radii=node_radii,
                                      eps=eps)
    edge_len = features['edge_len']
    strength = features['strength']
    thinness = features['thinness']
    misalign = features['misalign']

    # Soma roots and graph distance-to-soma
    root_nodes = cKDTree(node_centers).query(np.asarray(soma_mesh.vertices), k=1)[1]
    root_nodes = np.unique(np.asarray(root_nodes, dtype=int))
    if len(root_nodes) == 0:
        root_nodes = np.array([0], dtype=int)

    weights = loop_params.get('suspicion_weights', None)
    if weights is None:
        a = float(loop_params.get('a', 3.0))
        b = float(loop_params.get('b', 2.0))
        c = float(loop_params.get('c', 4.0))
    else:
        a = float(weights.get('a', 3.0))
        b = float(weights.get('b', 2.0))
        c = float(weights.get('c', 4.0))

    suspicion0 = a * thinness + b * misalign

    robust_dist = bool(loop_params.get('robust_dist', True))
    if robust_dist:
        dist_to_soma = compute_robust_dist_to_soma(
            G=G,
            root_nodes=root_nodes.tolist(),
            edge_len=edge_len,
            strength=strength,
            suspicion=suspicion0,
            dij_lambda=float(loop_params.get('dij_lambda', 2.0)),
            strength_gamma=float(loop_params.get('strength_gamma', 1.0)),
            eps=eps,
        )
    else:
        G.es['len'] = edge_len
        dist_mat = np.array(G.distances(source=root_nodes.tolist(), weights='len'))
        dist_to_soma = dist_mat.min(axis=0)

    du = dist_to_soma[el[:, 0]]
    dv = dist_to_soma[el[:, 1]]
    sideways = 1 - np.clip(np.abs(du - dv) / (edge_len + eps), 0, 1)
    suspicion = a * thinness + b * misalign + c * sideways

    cost_edge = (
        float(loop_params.get('w_len', 1.0)) * edge_len
        + float(loop_params.get('w_susp', 1.0)) * suspicion
        - float(loop_params.get('w_strength', 1.0)) * np.log(eps + strength)
    )

    if mode == 'soma_farthest':
        # near-first Kruskal key: (d_edge ASC, suspicion ASC, strength DESC, edge_len ASC)
        d_edge = np.maximum(du, dv)
        order = np.lexsort((edge_len, -strength, suspicion, d_edge))
        keep_ix = _kruskal_choose_edges(len(G.vs), el, order)
        return el[keep_ix]

    if mode != 'soma_rooted':
        raise ValueError(f'Unknown soma-aware loop break mode: "{mode}"')

    # Rooted mode: assign each node a parent towards soma
    edge_lookup = {tuple(sorted((int(u), int(v)))): i for i, (u, v) in enumerate(el)}
    roots = set(root_nodes.tolist())
    chosen = set()
    hard_thr = loop_params.get('suspicion_hard_thr', None)

    for v in np.argsort(-dist_to_soma):
        v = int(v)
        if v in roots:
            continue
        nbrs = np.array(G.neighbors(v), dtype=int)
        if len(nbrs) == 0:
            continue

        closer = nbrs[dist_to_soma[nbrs] < dist_to_soma[v]]
        if len(closer) == 0:
            closer = np.array([nbrs[np.argmin(dist_to_soma[nbrs])]], dtype=int)

        # Guardrail: if possible, avoid very suspicious closer-to-root edges.
        if hard_thr is not None and len(closer) > 1:
            closer_ix = np.array([edge_lookup.get(tuple(sorted((v, int(p)))), -1) for p in closer], dtype=int)
            valid = closer_ix >= 0
            if np.any(valid):
                valid_ix = closer_ix[valid]
                low_susp_valid = suspicion[valid_ix] <= float(hard_thr)
                if np.any(low_susp_valid):
                    keep = np.zeros_like(valid, dtype=bool)
                    keep[np.where(valid)[0][low_susp_valid]] = True
                    closer = closer[keep]

        best_parent = None
        best_cost = np.inf
        for p in closer:
            p = int(p)
            eix = edge_lookup.get(tuple(sorted((v, p))))
            if eix is None:
                continue
            cost = cost_edge[eix]
            if cost < best_cost:
                best_cost = cost
                best_parent = p

        if best_parent is not None:
            chosen.add(tuple(sorted((v, best_parent))))

    tree_edges = np.array(sorted(chosen), dtype=int)
    if bool(loop_params.get('repair_pass', False)) and len(tree_edges):
        tree_edges = np.array(
            repair_tree_local_swaps(
                G=G,
                tree_edges=[tuple(map(int, e)) for e in tree_edges.tolist()],
                cost_edge=cost_edge,
                topk=int(loop_params.get('repair_topk', 200)),
                max_hops=int(loop_params.get('repair_max_hops', 3)),
            ),
            dtype=int,
        )

    return tree_edges


def _compute_edge_features(el, node_centers, node_radii, eps=1e-12):
    """Compute edge features used in soma-aware loop handling."""
    edge_len = np.linalg.norm(node_centers[el[:, 0]] - node_centers[el[:, 1]], axis=1)
    vect, alpha = dotprops(node_centers)

    strength = np.vstack((node_radii[el[:, 0]], node_radii[el[:, 1]])).mean(axis=0)
    strength *= np.vstack((alpha[el[:, 0]], alpha[el[:, 1]])).mean(axis=0)
    if np.any(strength > 0):
        strength[strength <= 0] = strength[strength > 0].min() / 2
    else:
        strength[:] = 1

    ru = node_radii[el[:, 0]]
    rv = node_radii[el[:, 1]]
    thinness = 1 - (np.minimum(ru, rv) / np.maximum(np.maximum(ru, rv), eps))
    misalign = 1 - np.abs(np.sum(vect[el[:, 0]] * vect[el[:, 1]], axis=1))

    return {
        'el': el,
        'edge_len': edge_len,
        'strength': strength,
        'misalign': misalign,
        'thinness': thinness,
    }


def compute_robust_dist_to_soma(
    G: ig.Graph,
    root_nodes: list,
    edge_len: np.ndarray,
    strength: np.ndarray,
    suspicion: np.ndarray,
    dij_lambda: float = 2.0,
    strength_gamma: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute robust multi-source geodesic distance-to-soma on the contracted graph.

    Uses penalized Dijkstra weights:
        w_dij(e) = edge_len(e) * (1 + dij_lambda * suspicion(e)) / (eps + strength(e))**strength_gamma
    """
    w_dij = edge_len * (1 + dij_lambda * suspicion) / np.power(eps + strength, strength_gamma)
    G.es['w_dij'] = w_dij
    dist_mat = np.array(G.distances(source=root_nodes, weights='w_dij'))
    return dist_mat.min(axis=0)


def repair_tree_local_swaps(G: ig.Graph,
                            tree_edges: list,
                            cost_edge: np.ndarray,
                            topk: int = 200,
                            max_hops: int = 3) -> list:
    """Local post-pass to swap high-cost tree edges with nearby alternatives."""
    if len(tree_edges) == 0:
        return []

    all_edges = np.array(G.get_edgelist(), dtype=int)
    edge_idx = {tuple(sorted((int(u), int(v)))): i for i, (u, v) in enumerate(all_edges)}
    neighbors = [set(map(int, G.neighbors(i))) for i in range(len(G.vs))]

    tree_set = {tuple(sorted((int(u), int(v)))) for u, v in tree_edges}

    def bfs_limited(start, hops):
        visited = {int(start)}
        frontier = {int(start)}
        for _ in range(max(0, int(hops))):
            nxt = set()
            for n in frontier:
                nxt.update(neighbors[n])
            nxt -= visited
            if not nxt:
                break
            visited.update(nxt)
            frontier = nxt
        return visited

    def tree_components_without(rem_edge):
        a, b = rem_edge
        stack = [a]
        comp_a = {a}
        while stack:
            cur = stack.pop()
            for nb in neighbors[cur]:
                edge = tuple(sorted((cur, nb)))
                if edge == rem_edge or edge not in tree_set or nb in comp_a:
                    continue
                comp_a.add(nb)
                stack.append(nb)
        return comp_a

    ranked = sorted(tree_set,
                    key=lambda e: cost_edge[edge_idx[e]],
                    reverse=True)[:max(1, int(topk))]

    for rem_edge in ranked:
        if rem_edge not in tree_set:
            continue
        u, v = rem_edge
        comp_u = tree_components_without(rem_edge)
        if len(comp_u) == len(G.vs) or len(comp_u) == 0:
            continue

        uset = bfs_limited(u, max_hops)
        vset = bfs_limited(v, max_hops)

        best = None
        best_cost = np.inf
        for a in uset:
            for b in neighbors[a]:
                edge = tuple(sorted((a, b)))
                if edge in tree_set:
                    continue
                connects = (a in comp_u) != (b in comp_u)
                if not connects:
                    continue
                if (a not in vset) and (b not in vset):
                    continue
                c = cost_edge[edge_idx[edge]]
                if c < best_cost:
                    best = edge
                    best_cost = c

        if best is None:
            continue

        if best_cost < cost_edge[edge_idx[rem_edge]]:
            tree_set.remove(rem_edge)
            tree_set.add(best)

    return sorted(tree_set)


def _kruskal_choose_edges(n_nodes, edges, order):
    """Union-find Kruskal helper returning selected edge indices."""
    parent = np.arange(n_nodes, dtype=int)
    rank = np.zeros(n_nodes, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True

    chosen = []
    for idx in order:
        u, v = edges[int(idx)]
        if union(int(u), int(v)):
            chosen.append(int(idx))

    return np.array(chosen, dtype=int)


def _cast_waves(mesh, waves=1, origins=None, step_size=1,
                rad_agg_func=np.mean, progress=True, strict_origins=False):
    """Cast waves across mesh."""
    if not isinstance(origins, type(None)):
        if isinstance(origins, int):
            origins = [origins]
        elif not isinstance(origins, (set, list, np.ndarray)):
            raise TypeError('`origins` must be vertex ID (int) or list '
                            f'thereof, got "{type(origins)}"')
        origins = np.asarray(origins).astype(int)
    else:
        origins = np.array([])

    # Wave must be a positive integer >= 1
    waves = int(waves)
    if waves < 1:
        raise ValueError('`waves` must be integer >= 1')

    # Same for step size
    step_size = int(step_size)
    if step_size < 1:
        raise ValueError('`step_size` must be integer >= 1')

    # Generate Graph (must be undirected)
    G = ig.Graph(edges=mesh.edges_unique, directed=False)

    # Prepare empty array to fill with centers
    centers = np.full((mesh.vertices.shape[0], 3, waves), fill_value=np.nan)
    radii = np.full((mesh.vertices.shape[0], waves), fill_value=np.nan)

    # Go over each connected component
    with tqdm(desc='Skeletonizing', total=len(G.vs), disable=not progress) as pbar:
        for cc in G.connected_components():
            # Make a subgraph for this connected component
            SG = G.subgraph(cc)
            cc = np.array(cc)

            # Select seeds according to the number of waves
            n_waves = min(waves, len(cc))
            pot_seeds = np.arange(len(cc))
            np.random.seed(1985)  # make seeds predictable

            # See if we can use any origins
            if len(origins):
                # Get those origins in this cc
                in_cc = np.isin(origins, cc)
                if any(in_cc):
                    # Map origins into cc
                    cc_map = dict(zip(cc, np.arange(0, len(cc))))
                    seeds = np.array([cc_map[o] for o in origins[in_cc]])
                else:
                    seeds = np.array([])

                if len(seeds) < n_waves:
                    if strict_origins and len(seeds) > 0:
                        seeds = np.append(seeds,
                                          np.random.choice(seeds,
                                                           size=n_waves - len(seeds),
                                                           replace=True))
                    elif strict_origins and len(seeds) == 0:
                        n_waves = 1
                        seeds = np.array([pot_seeds[0]], dtype=int)
                    else:
                        remaining_seeds = pot_seeds[~np.isin(pot_seeds, seeds)]
                        seeds = np.append(seeds,
                                          np.random.choice(remaining_seeds,
                                                           size=n_waves - len(seeds),
                                                           replace=False))
            else:
                seeds = np.random.choice(pot_seeds, size=n_waves, replace=False)

            seeds = seeds.astype(int)

            # Get the distance between the seeds and all other nodes
            dist = np.array(SG.distances(source=seeds, target=None, mode='all'))

            if step_size > 1:
                mx = dist.flatten()
                mx = mx[mx < float('inf')].max()
                dist = np.digitize(dist, bins=np.arange(0, mx, step_size))

            # Cast the desired number of waves
            for w in range(dist.shape[0]):
                this_wave = dist[w, :]
                # Collect groups
                mx = this_wave[this_wave < float('inf')].max()
                for i in range(0, int(mx) + 1):
                    this_dist = this_wave == i
                    ix = np.where(this_dist)[0]
                    SG2 = SG.subgraph(ix)
                    for cc2 in SG2.connected_components():
                        this_verts = cc[ix[cc2]]
                        this_center = mesh.vertices[this_verts].mean(axis=0)
                        this_radius = cdist(this_center.reshape(1, -1), mesh.vertices[this_verts])
                        this_radius = rad_agg_func(this_radius)
                        centers[this_verts, :, w] = this_center
                        radii[this_verts, w] = this_radius

            pbar.update(len(cc))

    # Get mean centers and radii over all the waves we casted
    centers_final = np.nanmean(centers, axis=2)
    radii_final = np.nanmean(radii, axis=1)

    return centers_final, radii_final, G


def dotprops(x, k=20):
    """Generate vectors and alpha from local neighborhood."""
    # Checks and balances
    n_points = x.shape[0]

    # Make sure we don't ask for more nearest neighbors than we have points
    k = min(n_points, k)

    # Create the KDTree and get the k-nearest neighbors for each point
    tree = cKDTree(x)
    dist, ix = tree.query(x, k=k)
    # This makes sure we have (N, k) shaped array even if k = 1
    ix = ix.reshape(x.shape[0], k)

    # Get points: array of (N, k, 3)
    pt = x[ix]

    # Generate centers for each cloud of k nearest neighbors
    centers = np.mean(pt, axis=1)

    # Generate vector from center
    cpt = pt - centers.reshape((pt.shape[0], 1, 3))

    # Get inertia (N, 3, 3)
    inertia = cpt.transpose((0, 2, 1)) @ cpt

    # Extract vector and alpha
    u, s, vh = np.linalg.svd(inertia)
    vect = vh[:, 0, :]
    alpha = (s[:, 0] - s[:, 1]) / np.sum(s, axis=1)

    return vect, alpha


def _selftest_keep_loops():
    """Lightweight sanity checks for loop-preserving wavefront return path."""
    edges_cycle = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
    node_centers = np.array([[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]])
    node_radii = np.array([1.0, 1.0, 1.0])
    mesh_map = np.array([0, 1, 2], dtype=int)
    G_cycle = ig.Graph(edges=edges_cycle, directed=False)

    mesh_stub = type('MeshStub', (), {
        'vertices': node_centers,
        'faces': np.array([[0, 1, 2]], dtype=int)
    })()

    original_contract = _wavefront_contracted_graph
    original_make_trimesh = make_trimesh
    try:
        def _fake_contracted_graph(**kwargs):
            return node_centers, node_radii, G_cycle.copy(), mesh_map

        def _fake_make_trimesh(mesh, validate=False):
            return mesh

        globals()['_wavefront_contracted_graph'] = _fake_contracted_graph
        globals()['make_trimesh'] = _fake_make_trimesh

        out_edges, out_vertices = by_wavefront_keep_loops(mesh_stub,
                                                          waves=1,
                                                          progress=False)
        assert isinstance(out_edges, np.ndarray)
        assert isinstance(out_vertices, np.ndarray)
        assert out_edges.shape[1] == 2
        assert out_vertices.shape == node_centers.shape

        expected = {tuple(sorted(e)) for e in edges_cycle.tolist()}
        observed = {tuple(sorted(e)) for e in out_edges.tolist()}
        assert observed == expected
        assert np.allclose(out_vertices, node_centers)
    finally:
        globals()['_wavefront_contracted_graph'] = original_contract
        globals()['make_trimesh'] = original_make_trimesh

    return True


def _self_test_soma_loop_break():
    """Lightweight sanity checks for soma-aware cycle breaking."""
    # Near-soma suspicious shortcut: edge (1, 2) links equal-distance nodes.
    centers_near = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
    ])
    radii_near = np.array([2.0, 2.0, 2.0, 1.0])
    edges_near = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    G_near = ig.Graph(edges=edges_near, directed=False)

    soma_mesh_near = type('SomaMeshNear', (), {'vertices': centers_near[[0]]})
    tree_near = _soma_aware_tree_edges(G_near, centers_near, radii_near,
                                       soma_mesh_near, 'soma_farthest',
                                       {'a': 3.0, 'b': 2.0, 'c': 8.0})
    near_set = {tuple(sorted(e)) for e in tree_near.tolist()}
    assert (1, 2) not in near_set

    # Far cycle should be broken at the far shortcut edge.
    centers_far = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],
        [11.0, 0.1, 0.0],
    ])
    radii_far = np.array([2.0, 2.0, 2.0, 1.8, 1.2, 1.2, 0.2])
    edges_far = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4],
        [4, 5], [5, 6], [4, 6],
    ])
    G_far = ig.Graph(edges=edges_far, directed=False)
    soma_mesh_far = type('SomaMeshFar', (), {'vertices': centers_far[[0, 1]]})

    tree_far = _soma_aware_tree_edges(G_far, centers_far, radii_far,
                                      soma_mesh_far, 'soma_farthest',
                                      {'robust_dist': True})
    assert len(tree_far) == len(centers_far) - 1

    tree_rooted = _soma_aware_tree_edges(G_far, centers_far, radii_far,
                                         soma_mesh_far, 'soma_rooted',
                                         {'robust_dist': True})
    rooted_set = {tuple(sorted(e)) for e in tree_rooted.tolist()}
    assert any(0 in e or 1 in e for e in rooted_set)

    return True


def _selftest():
    """Synthetic regression check for multiple shortcut false-mergers."""
    # Main chain 0-1-2-3-4-5 with two shortcut links (1-4) and (2-5).
    centers = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    radii = np.array([2.5, 2.2, 2.0, 1.8, 1.6, 1.4])
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
        [1, 4], [2, 5],
    ])
    G = ig.Graph(edges=edges, directed=False)
    soma_mesh = type('SomaMesh', (), {'vertices': centers[[0]]})

    params = {
        'robust_dist': True,
        'dij_lambda': 3.0,
        'strength_gamma': 1.0,
        'suspicion_weights': {'a': 3.0, 'b': 2.0, 'c': 4.0},
        'suspicion_hard_thr': 6.0,
    }
    tree = _soma_aware_tree_edges(G, centers, radii, soma_mesh, 'soma_rooted', params)
    tree_set = {tuple(sorted(e)) for e in tree.tolist()}

    chain_edges = {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}
    shortcut_edges = {(1, 4), (2, 5)}

    assert chain_edges.issubset(tree_set), f'Chain split detected: {tree_set}'
    assert tree_set.isdisjoint(shortcut_edges), f'Shortcut retained unexpectedly: {tree_set}'
    return True


if __name__ == '__main__':
    assert _self_test_soma_loop_break()
    assert _selftest()
    print('wave.py self-tests passed')
