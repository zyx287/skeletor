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

__all__ = ['make_trimesh', 'find_closest_node_to_centroid']

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

def find_closest_node_to_centroid(skeleton, mesh):
    """Find the node in a skeleton that is closest to the centroid of a mesh.
    
    This is particularly useful for finding a good root node for skeletonization
    algorithms that require a starting point, especially when working with neuronal
    data where the soma/nucleus is a natural starting point.
    
    Parameters
    ----------
    skeleton : skeletor.Skeleton
               The skeleton to find a node in.
    mesh :     trimesh.Trimesh
               Mesh representing the region of interest (e.g., nucleus or soma).
               The centroid of this mesh will be used as the reference point.
    
    Returns
    -------
    int
        Node ID of the skeleton node closest to the mesh centroid.
    
    Examples
    --------
    >>> import skeletor as sk
    >>> import trimesh
    >>> mesh = sk.example_mesh()
    >>> skel = sk.skeletonize.by_wavefront(mesh)
    >>> soma_mesh = trimesh.primitives.Sphere(radius=500, center=[10000, 20000, 15000])
    >>> root_node = sk.utilities.find_closest_node_to_centroid(skel, soma_mesh)
    >>> # Use this node as the root for other operations
    >>> skel = skel.reroot(root_node)
    """
    mesh = make_trimesh(mesh, validate=False)
    centroid = mesh.centroid

    node_coords = skeleton.swc[['x', 'y', 'z']].values

    distances = np.sqrt(np.sum((node_coords - centroid)**2, axis=1))

    closest_node_index = np.argmin(distances)
    closest_node_id = skeleton.swc.iloc[closest_node_index].node_id
    
    return int(closest_node_id)