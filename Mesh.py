import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class Mesh:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            nnodes, ntri, nquad, _, _, _, _ = map(int, f.readline().split())
            coords = [list(map(float, f.readline().split())) for _ in range(nnodes)]
            self.coords = np.array(coords)  # (nnodes,3)

            tris = np.array([list(map(int, f.readline().split())) for _ in range(ntri)], dtype=int) - 1 if ntri > 0 else np.zeros((0,3), dtype=int)
            quads = np.array([list(map(int, f.readline().split())) for _ in range(nquad)], dtype=int) - 1 if nquad > 0 else np.zeros((0,4), dtype=int)

            # unify into one (nelem,4) array
            self.elems = np.full((ntri+nquad, 4), -1, dtype=int)
            if ntri > 0:
                self.elems[:ntri, :3] = tris
            if nquad > 0:
                self.elems[ntri:, :] = quads
            self.n_owned=self.elems.shape[0]

    def info(self):
        print(f"Mesh info: nnodes={self.coords.shape[0]}, nelem={self.elems.shape[0]}")
        print(f"Coordinates shape: {self.coords.shape}")
        print(f"elems shape: {self.elems.shape}")

    def plot2d(self, ax=None, show=True):
        """
        Plot the mesh in 2D using matplotlib.
        Halo cells are shown in red, owned cells in black.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        if ax is None:
            fig, ax = plt.subplots()
        n_owned = getattr(self, 'n_owned', None)
        if n_owned is None:
            # if not set, assume all elements are owned
            n_owned = self.elems.shape[0]
        # owned elements
        polys_owned = []
        for elem in self.elems[:n_owned]:
            verts = elem[elem != -1]
            polys_owned.append(self.coords[verts, :2])
        pc_owned = PolyCollection(polys_owned, edgecolors='k', facecolors='none')
        ax.add_collection(pc_owned)

        #halo elements
        if n_owned < self.elems.shape[0]:
            polys_halo = []
            for elem in self.elems[n_owned:]:
                verts = elem[elem != -1]
                polys_halo.append(self.coords[verts, :2])
            pc_halo = PolyCollection(polys_halo, edgecolors='r', facecolors='none', linestyle='--')
            ax.add_collection(pc_halo)
        ax.autoscale()
        ax.set_aspect('equal')
        if show:
            plt.show()
        return ax
    
    def remove_duplicate_nodes_tol(self, tol=1e-12):
        """
        Remove duplicate nodes using a floating-point tolerance.
        Preserves -1 padded entries in self.elems.
        Keeps the original order of nodes.
        """
        coords = self.coords
        elems = self.elems

        # Round coordinates to integers based on tolerance
        coords_rounded = np.round(coords / tol).astype(np.int64)

        # Get unique coordinates and inverse mapping, preserving first occurrence order
        coords_view = np.ascontiguousarray(coords_rounded).view(np.dtype((np.void, coords_rounded.dtype.itemsize * coords_rounded.shape[1])))
        _, idx, inverse = np.unique(coords_view, return_index=True, return_inverse=True)
        idx_sort = np.argsort(idx)
        coords_unique = coords[ idx[idx_sort] ]

        # Remap inverse to preserve original order
        inv_map = np.zeros_like(inverse)
        for new_idx, old_idx in enumerate(idx_sort):
            inv_map[inverse == old_idx] = new_idx

        # Update element connectivity
        elems_new = np.full_like(elems, -1)
        for i in range(elems.shape[0]):
            for j in range(elems.shape[1]):
                n = elems[i,j]
                if n != -1:
                    elems_new[i,j] = inv_map[n]

        self.elems = elems_new
        self.coords = coords_unique

    def remove_duplicate_nodes_cantor(self, scale=1e6):
        """
        Remove duplicate nodes using Cantor pairing (no floating-point rounding issues).
        Preserves -1 padded entries in elems.
        scale: scale factor to convert float coordinates to integers.
        """
        coords = self.coords
        elems = self.elems

        # scale and convert to integers
        coords_int = np.round(coords[:, :3] * scale).astype(np.int64)

        # Cantor pairing function
        def cantor_pair(a, b):
            return (a + b) * (a + b + 1) // 2 + b

        # pack 3D coords into single integer
        packed = cantor_pair(cantor_pair(coords_int[:,0], coords_int[:,1]), coords_int[:,2])

        # get unique nodes and inverse mapping
        unique_packed, inverse = np.unique(packed, return_inverse=True)

        # update element connectivity
        elems_new = np.full_like(elems, -1)
        for i in range(elems.shape[0]):
            for j in range(elems.shape[1]):
                n = elems[i,j]
                if n != -1:
                    elems_new[i,j] = inverse[n]

        self.elems = elems_new
        self.coords = coords[inverse[:len(coords)]]  # keep original floats for unique nodes

    def build_edge_to_elem(self):
        # TODO this will not work yet with quads
        edges = []           # Store edges in original order
        edge_lookup = {}     # Map sorted vertex pair -> edge index
        edge2elem = []
        elem2edge = [[] for _ in range(len(self.elems))]

        for eidx, elem in enumerate(self.elems):
            valid = elem[elem != -1]
            n = len(valid)
            for i in range(n):
                a, b = valid[i], valid[(i + 1) % n]
                key = tuple(sorted((a, b)))  # for lookup
                if key not in edge_lookup:
                    edge_lookup[key] = len(edges)
                    edges.append((a, b))     # store in original order
                    edge2elem.append([-1, -1])
                edid = edge_lookup[key]
                if edge2elem[edid][0] == -1:
                    edge2elem[edid][0] = eidx
                else:
                    edge2elem[edid][1] = eidx
                elem2edge[eidx].append(edid)

        edges_array = np.array(edges, dtype=int)
        edge2elem = np.array(edge2elem, dtype=int)
        return edges_array, edge2elem, np.array(elem2edge)

    def add_halo_cells(self):
        """
        Add halo/ghost elements:
          1) Edge-based halos for each boundary edge (reuse edge nodes; reflect others).
          2) Node-based ghosts for elements that touch wall nodes but have no boundary edge
             (uses average normal of incident boundary edges on each wall node).
        Corner fixes:
           1) elements that have two wall nodes whose connecting edge is NOT a boundary edge.
           2) elements that have two boundary edges at corners

        Returns
        -------
        ghost_map : (n_new_ghosts,) int
            Mapping ghost index -> owner (real) element index (in the pre-append indexing).
        """
        import numpy as np

        # Save owned count once (used by plotting, etc.)
        if not hasattr(self, "n_owned"):
            self.n_owned = self.elems.shape[0]
        start_owned_elems = self.elems.shape[0]
        start_nodes = self.coords.shape[0]

        # --- Build edge structures and detect boundary edges ---
        edges, edge2elem, _ = self.build_edge_to_elem()
        bnd_edge_ids = np.where(edge2elem[:, 1] == -1)[0]
        # Unordered pairs for quick membership test
        bnd_edge_set = {tuple(sorted((edges[i,0], edges[i,1]))) for i in bnd_edge_ids}

        # Collect wall nodes and per-node boundary-edge normals
        wall_nodes_set = set()
        node_normals = {}  # node -> list of normals (unit 2D)

        def edge_unit_normal(p1, p2):
            v = p2 - p1
            nv = np.linalg.norm(v)
            if nv == 0.0:
                return np.array([0.0, 0.0])
            v /= nv
            n = np.array([-v[1], v[0]])  # either outward/inward—sign does not matter for reflection
            return n

        # --- Step 1: edge-based halos ---
        new_coords = []
        new_elems = []
        ghost_map = []

        for eid in bnd_edge_ids:
            left_elem = edge2elem[eid, 0]
            elem_nodes = self.elems[left_elem]
            valid = elem_nodes[elem_nodes != -1]
            elem_xy = self.coords[valid, :2]

            n1, n2 = edges[eid]
            wall_nodes_set.update((n1, n2))

            p1, p2 = self.coords[n1, :2], self.coords[n2, :2]
            n = edge_unit_normal(p1, p2)

            # record normals for each wall node on this boundary edge
            for nd in (n1, n2):
                node_normals.setdefault(nd, []).append(n)

            halo_nodes = []
            for i, nd in enumerate(valid):
                p = elem_xy[i]
                if nd == n1 or nd == n2:
                    halo_nodes.append(nd)  # reuse boundary-edge node
                else:
                    # reflect interior vertex across this boundary edge
                    d = np.dot(p - p1, n)
                    pref = p - 2.0 * d * n
                    new_coords.append([pref[0], pref[1], 0.0])
                    halo_nodes.append(start_nodes + len(new_coords) - 1)

            padded = np.full(4, -1, dtype=int)
            padded[:len(halo_nodes)] = halo_nodes
            new_elems.append(padded)
            ghost_map.append(left_elem)

        wall_nodes = np.array(sorted(wall_nodes_set), dtype=int)

        # Average normal per wall node (unit length)
        avg_normal = {}
        for nd in wall_nodes:
            arr = np.asarray(node_normals.get(nd, []), dtype=float)
            if arr.size == 0:
                avg_normal[nd] = np.array([0.0, 0.0])
            else:
                nsum = arr.sum(axis=0)
                nn = np.linalg.norm(nsum)
                avg_normal[nd] = nsum / nn if nn > 0 else np.array([0.0, 0.0])

        # Precompute: does an element have at least one boundary edge?
        has_bnd_edge = np.zeros(start_owned_elems, dtype=int)
        for eidx in range(start_owned_elems):
            valid = self.elems[eidx][self.elems[eidx] != -1]
            m = len(valid)
            for i in range(m):
                a, b = valid[i], valid[(i + 1) % m]
                if tuple(sorted((a, b))) in bnd_edge_set:
                    has_bnd_edge[eidx] +=1

        # --- Step 2: node-based ghosts (for elems touching wall nodes, but no boundary edge) ---
        for eidx in range(start_owned_elems):
            # skip if already handled by edge-based halos (has one boundary edge)
            # having more than 1 boundary edge is special case handled below
            if has_bnd_edge[eidx]==1:
                continue

            elem_nodes = self.elems[eidx]
            valid = elem_nodes[elem_nodes != -1]
            # must touch at least one wall node
            wall_in_elem = [nd for nd in valid if nd in wall_nodes_set]
            if not wall_in_elem:
                continue
            
            # remove this since we are dealing with this situtation within the loop
            # Corner fix: if TRIANGLE with exactly two wall nodes whose edge is NOT
            # a boundary edge -> skip
            #if len(valid) == 3 and len(wall_in_elem) == 2:
            #    pair = tuple(sorted((wall_in_elem[0], wall_in_elem[1])))
            #    if pair not in bnd_edge_set:
            #        # this is the corner-triangle case: do NOT create a ghost
            #        continue

            elem_xy = self.coords[valid, :2]

            # Build ghost: reuse wall nodes, reflect non-wall nodes using average of
            # neighboring wall-node normals
            # anchor point for reflection: average position of wall nodes in this element
            k=-1
            for w in wall_in_elem:
                ghost_nodes = []
                k+=1
                p0 = self.coords[np.array(w), :2]
                if (len(wall_in_elem) == 3) :
                    p1=wall_in_elem[(k+1)%3]
                    p2=wall_in_elem[(k+2)%3]
                    if tuple(sorted((p1, p2))) in bnd_edge_set:
                        continue
                # average normal from the wall nodes present in this element
                n_avg = avg_normal[w]
                #ns = np.array([avg_normal[nd] for nd in wall_in_elem], dtype=float)
                #nsum = ns.sum(axis=0)
                #ln = np.linalg.norm(nsum)
                #n_avg = nsum / ln if ln > 0 else np.array([0.0, 0.0])            
                
                for i, nd in enumerate(valid):
                    p = elem_xy[i]
                    if nd in wall_nodes_set and len(wall_in_elem)==1:
                        ghost_nodes.append(nd)  # reuse wall node
                    else:
                        # reflect across line through p0 with normal n_avg
                        d = np.dot(p - p0, n_avg)
                        pref = p - 2.0 * d * n_avg
                        new_coords.append([pref[0], pref[1], 0.0])
                        ghost_nodes.append(start_nodes + len(new_coords) - 1)

                padded = np.full(4, -1, dtype=int)
                padded[:len(ghost_nodes)] = ghost_nodes
                new_elems.append(padded)
                ghost_map.append(eidx)

        # --- Append new data (if any) ---
        if new_coords:
            self.coords = np.vstack([self.coords, np.array(new_coords, dtype=float)])
            self.elems = np.vstack([self.elems, np.array(new_elems, dtype=int)])

        return np.array(ghost_map, dtype=int)

    def build_elem2elem(self):
        """
        Build element-to-element connectivity (edge neighbors only).
        Returns a (nelem,4) array with -1 padding.
        For triangles, only first 3 entries are valid.
        """
        nelem = self.elems.shape[0]
        elem2elem = -np.ones((nelem, 4), dtype=int)

        # edge -> [element, local_edge_index]
        edges = {}
        for eidx, elem in enumerate(self.elems):
            valid = elem[elem != -1]
            n = len(valid)
            for i in range(n):
                a, b = valid[i], valid[(i+1) % n]
                edge = tuple(sorted((a, b)))
                if edge in edges:
                    other_eidx, other_lidx = edges[edge]
                    # connect both elements
                    elem2elem[eidx, i] = other_eidx
                    elem2elem[other_eidx, other_lidx] = eidx
                else:
                    edges[edge] = (eidx, i)

        return elem2elem

            
    # ---------- element-to-element (edge adjacency, FN1) ----------
    def build_elem2elem_csr(self):
        """Return element-to-element adjacency in CSR format (via edges)."""
        edges = defaultdict(list)   # edge -> [elem1, elem2]
        for eidx, elem in enumerate(self.elems):
            valid = elem[elem != -1]
            n = len(valid)
            for i in range(n):
                a, b = valid[i], valid[(i+1)%n]
                edge = tuple(sorted((a,b)))
                edges[edge].append(eidx)

        neighbors = [[] for _ in range(len(self.elems))]
        for adj in edges.values():
            if len(adj) == 2:
                e1, e2 = adj
                neighbors[e1].append(e2)
                neighbors[e2].append(e1)

        # convert to CSR
        indptr = [0]
        indices = []
        for nlist in neighbors:
            indices.extend(nlist)
            indptr.append(len(indices))
        return np.array(indices, dtype=int), np.array(indptr, dtype=int)

    def build_node2elem_dense_vec(self, xp=np):
        """
        Return node-to-element adjacency as a dense array (vectorized, no Python loops).

        Parameters
        ----------
        xp : module
            Backend array module (numpy or cupy).

        Returns
        -------
        indices : (nnodes, max_degree) int array
            Element indices connected to each node, padded with -1.
        mask : (nnodes, max_degree) bool array
            Boolean mask: True if entry is valid, False if padded.
        """
        elems = xp.array(self.elems, dtype=int)
        Ne, n_nodes = elems.shape
        nnodes = self.coords.shape[0]

        # Flatten element-node pairs, ignoring padding (-1)
        mask_valid = elems != -1
        flat_nodes = elems[mask_valid]
        flat_elems = xp.repeat(xp.arange(Ne), n_nodes)[mask_valid.ravel()]

        # Count neighbors per node
        counts = xp.bincount(flat_nodes, minlength=nnodes)
        max_degree = int(counts.max().get() if xp is not np else counts.max())

        # Sort flat_nodes to group identical nodes together
        order = xp.argsort(flat_nodes, kind="stable")
        sorted_nodes = flat_nodes[order]
        sorted_elems = flat_elems[order]

        # Assign positions within each node’s group
        pos_in_group = xp.arange(len(sorted_nodes)) - xp.concatenate(
            ([0], xp.cumsum(counts[:-1]))
        )[sorted_nodes]

        # Allocate dense arrays
        indices = xp.full((nnodes, max_degree), -1, dtype=int)
        mask = xp.zeros((nnodes, max_degree), dtype=bool)

        # Fill
        indices[sorted_nodes, pos_in_group] = sorted_elems
        mask[sorted_nodes, pos_in_group] = True

        return indices, mask

    # ---------- node-to-element adjacency ----------
    def build_node2elem(self):
        """Return node-to-element adjacency in CSR format."""
        node2elems = defaultdict(list)
        for eidx, elem in enumerate(self.elems):
            for n in elem:
                if n != -1:
                    node2elems[n].append(eidx)

        nnodes = self.coords.shape[0]
        indptr = [0]
        indices = []
        for n in range(nnodes):
            indices.extend(node2elems[n])
            indptr.append(len(indices))
        return np.array(indices, dtype=int), np.array(indptr, dtype=int)
    
    # ---------- elem2elem fn2 adjacency ----------
    def build_elem2elem_fn2_csr(self):
        """
        Build FN2 connectivity (edge + vertex neighbors).
        Returns CSR representation: indptr, indices, data
        - indptr: row pointers (Ne+1,)
        - indices: neighbor element indices
        - data: all ones (for adjacency)
        """
        Ne = self.elems.shape[0]
        node2elem = [[] for _ in range(self.coords.shape[0])]

        # Build node-to-element map
        for e, elem in enumerate(self.elems):
            for n in elem:
                if n != -1:
                    node2elem[n].append(e)

        # Collect neighbors
        neighbors = [[] for _ in range(Ne)]
        for e, elem in enumerate(self.elems):
            neigh = set()
            for n in elem:
                if n != -1:
                    neigh.update(node2elem[n])
            neigh.discard(e)  # remove self
            neighbors[e] = sorted(neigh)

        # Convert to CSR
        indptr = np.zeros(Ne + 1, dtype=np.int32)
        indices = []
        data = []
        count = 0
        for e in range(Ne):
            indptr[e] = count
            indices.extend(neighbors[e])
            data.extend([1] * len(neighbors[e]))
            count += len(neighbors[e])
        indptr[Ne] = count

        return (
            np.array(indptr, dtype=np.int32),
            np.array(indices, dtype=np.int32),
            np.array(data, dtype=np.int8),
        )

    def compute_centroids_and_areas(self):
        """
        Compute centroids and areas of all elements (triangles/quads/halos).
        - For triangles: arithmetic mean of vertices; area by shoelace formula.
        - For quads: area-weighted centroid using two triangles; area is sum.

        Returns:
            centroids : (Ne, 2) array of (x, y) centroids
            areas     : (Ne,) array of element areas
        """
        Ne = self.elems.shape[0]
        centroids = np.zeros((Ne, 2))
        areas = np.zeros(Ne)

        def tri_area_and_centroid(pts):
            # shoelace area
            area = 0.5 * abs(
                pts[0, 0] * (pts[1, 1] - pts[2, 1])
                + pts[1, 0] * (pts[2, 1] - pts[0, 1])
                + pts[2, 0] * (pts[0, 1] - pts[1, 1])
            )
            centroid = pts.mean(axis=0)
            return area, centroid

        for i, elem in enumerate(self.elems):
            valid = elem[elem != -1]  # strip padding
            coords = self.coords[valid, :2]

            if len(valid) == 3:
                A, C = tri_area_and_centroid(coords)
                areas[i] = A
                centroids[i] = C

            elif len(valid) == 4:
                tri1 = coords[[0, 1, 2]]
                tri2 = coords[[0, 2, 3]]

                A1, C1 = tri_area_and_centroid(tri1)
                A2, C2 = tri_area_and_centroid(tri2)

                areas[i] = A1 + A2
                centroids[i] = (A1 * C1 + A2 * C2) / areas[i]

            else:
                raise ValueError(f"Unsupported element with {len(valid)} nodes")

        return centroids, areas
