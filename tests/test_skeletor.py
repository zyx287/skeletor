import skeletor as sk
import trimesh as tm
import networkx as nx
import numpy as np
import ncollpyde


def _count_crossing_edges(skeleton, mesh, eps=1e-6):
    """Count skeleton edges that cross the mesh boundary."""
    coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)
    swc = skeleton.swc

    not_root = swc.parent_id >= 0
    if not np.any(not_root):
        return 0

    sources = swc.loc[not_root, ['x', 'y', 'z']].values
    parent_ids = swc.loc[not_root, 'parent_id'].values
    targets = swc.set_index('node_id').loc[parent_ids, ['x', 'y', 'z']].values

    ix, loc, _ = coll.intersections(sources, targets)
    if len(ix) == 0:
        return 0

    d_src = np.linalg.norm(loc - sources[ix], axis=1)
    d_tgt = np.linalg.norm(loc - targets[ix], axis=1)
    crossing = (d_src > eps) & (d_tgt > eps)
    if not crossing.any():
        return 0

    return int(np.unique(ix[crossing]).size)


class TestPreprocessing:
    def test_example_mesh(self):
        """Test loading the example mesh."""
        assert isinstance(sk.example_mesh(), tm.Trimesh)

    def test_fix_mesh(self):
        fixed = sk.pre.fix_mesh(sk.example_mesh(),
                                remove_disconnected=True,  # this is off by default
                                inplace=False)
        assert isinstance(fixed, tm.Trimesh)

    def test_contraction(self):
        cont = sk.pre.contract(sk.example_mesh(), epsilon=0.1)
        assert isinstance(cont, tm.Trimesh)


class TestSkeletonization:
    def test_wavefront_default_backward_compatibility(self):
        s = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=1)
        assert isinstance(s.vertices, np.ndarray)
        assert isinstance(s.edges, np.ndarray)
        assert isinstance(s.mesh_map, np.ndarray)
        assert s.vertices.shape[1] == 3
        assert s.edges.shape[1] == 2
        assert len(s.mesh_map) == len(s.mesh.vertices)

    def test_wave_skeletonization(self):
        one_wave = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=1)
        two_wave = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=2)
        stepsize = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=1,
                                               step_size=2)

        for s in [one_wave, two_wave, stepsize]:
            assert len(s.mesh_map) == len(s.mesh.vertices)
            assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_wavefront_inside_nodes_mode(self):
        mesh = sk.pre.fix_mesh(sk.example_mesh(), remove_disconnected=5, inplace=False)
        s = sk.skeletonize.by_wavefront(mesh,
                                        waves=1,
                                        inside_mode='nodes',
                                        center_mode='inside_mean',
                                        progress=False)
        coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)
        assert coll.contains(s.vertices).all()

    def test_wavefront_inside_nodes_edges_mode(self):
        mesh = sk.pre.fix_mesh(sk.example_mesh(), remove_disconnected=5, inplace=False)
        s = sk.skeletonize.by_wavefront(mesh,
                                        waves=1,
                                        inside_mode='nodes_edges',
                                        center_mode='inside_mean',
                                        progress=False)
        coll = ncollpyde.Volume(mesh.vertices, mesh.faces, validate=False)
        assert coll.contains(s.vertices).all()
        assert _count_crossing_edges(s, mesh) == 0

    def test_wavefront_strict_can_add_nodes(self):
        mesh = sk.pre.fix_mesh(sk.example_mesh(), remove_disconnected=5, inplace=False)
        default = sk.skeletonize.by_wavefront(mesh, waves=1, progress=False)
        strict = sk.skeletonize.by_wavefront(mesh,
                                             waves=1,
                                             inside_mode='nodes_edges',
                                             center_mode='inside_mean',
                                             progress=False)
        assert strict.swc.shape[0] >= default.swc.shape[0]

    def test_wavefront_max_edge_fix_iter_zero_does_not_crash(self):
        mesh = sk.pre.fix_mesh(sk.example_mesh(), remove_disconnected=5, inplace=False)
        s = sk.skeletonize.by_wavefront(mesh,
                                        waves=1,
                                        inside_mode='nodes_edges',
                                        center_mode='inside_mean',
                                        max_edge_fix_iter=0,
                                        progress=False)
        assert isinstance(s.vertices, np.ndarray)
        assert isinstance(s.edges, np.ndarray)
        assert isinstance(s.mesh_map, np.ndarray)

    def test_vertex_cluster(self):
        s = sk.skeletonize.by_vertex_clusters(sk.example_mesh(),
                                              sampling_dist=100)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_edge_collapse(self):
        s = sk.skeletonize.by_edge_collapse(sk.example_mesh())

    def test_teasar(self):
        s = sk.skeletonize.by_teasar(sk.example_mesh(), 500)

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_tangent(self):
        s = sk.skeletonize.by_tangent_ball(sk.example_mesh())

        assert len(s.mesh_map) == len(s.mesh.vertices)
        assert all(np.isin(s.mesh_map, s.swc.node_id.values))

    def test_graph(self):
        s = sk.skeletonize.by_wavefront(sk.example_mesh(), waves=1)

        assert isinstance(s.get_graph(), nx.DiGraph)


class TestPostprocessing:
    def test_cleanup(self):
        mesh = sk.example_mesh()
        s = sk.skeletonize.by_wavefront(mesh, waves=1)
        clean = sk.post.clean_up(s, inplace=False)

    def test_radius(self):
        mesh = sk.example_mesh()
        s = sk.skeletonize.by_wavefront(mesh, waves=1)

        rad_knn = sk.post.radii(s, method='knn')
        rad_ray = sk.post.radii(s, method='ray')


class TestExamples:
    def test_readme_example(self):
        mesh = sk.example_mesh()
        fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
        skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)

        assert isinstance(skel.vertices, np.ndarray)
        assert isinstance(skel.edges, np.ndarray)
        assert isinstance(skel.mesh_map, np.ndarray)
        assert skel.vertices.shape[1] == 3
        assert skel.edges.shape[1] == 2
        assert skel.mesh_map.shape[0] == len(skel.mesh.vertices)
