#!/usr/bin/env python3
import os
import sys
import time
import heapq
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import scipy
import trimesh
from skimage.measure import marching_cubes

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# ==========================================
# 1. Mesh & Test Case Generators
# ==========================================

def make_open_cylinder(radius=1.0, height=2.0, sections=32, segments=16):
    """Generates a parametric open-top and open-bottom cylinder."""
    vertices = []
    for j in range(segments + 1):
        z = -height/2.0 + height * (j / segments)
        for i in range(sections):
            theta = 2.0 * np.pi * i / sections
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            vertices.append([x, y, z])
    vertices = np.array(vertices)
    
    faces = []
    for j in range(segments):
        for i in range(sections):
            next_i = (i + 1) % sections
            v00 = j * sections + i
            v10 = j * sections + next_i
            v01 = (j + 1) * sections + i
            v11 = (j + 1) * sections + next_i
            
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
            
    faces = np.array(faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def make_test_sphere(radius=1.0, subdivisions=3):
    """Generates a subdivision sphere using trimesh."""
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)

# ==========================================
# 2. Triangle Distance Projection
# ==========================================

def closest_point_on_triangle(A, B, C, P):
    """Calculates the exact closest point on a 3D triangle to a point P."""
    AB = B - A
    AC = C - A
    AP = P - A
    
    d1 = np.dot(AB, AP)
    d2 = np.dot(AC, AP)
    if d1 <= 0.0 and d2 <= 0.0:
        return A
        
    BP = P - B
    d3 = np.dot(AB, BP)
    d4 = np.dot(AC, BP)
    if d3 >= 0.0 and d4 <= d3:
        return B
        
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return A + v * AB
        
    CP = P - C
    d5 = np.dot(AB, CP)
    d6 = np.dot(AC, CP)
    if d6 >= 0.0 and d5 <= d6:
        return C
        
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return A + w * AC
        
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return B + w * (C - B)
        
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return A + v * AB + w * AC

def project_point_to_mesh(mesh, P):
    """Projects a 3D point P onto the mesh surface, returning coords and face index."""
    sq_dists = np.sum((mesh.vertices - P)**2, axis=1)
    v = np.argmin(sq_dists)
    
    # Check triangles adjacent to closest vertex
    face_indices = np.where(mesh.faces == v)[0]
    best_pt = None
    best_dist_sq = np.inf
    best_face = -1
    
    for f_idx in face_indices:
        f = mesh.faces[f_idx]
        A = mesh.vertices[f[0]]
        B = mesh.vertices[f[1]]
        C = mesh.vertices[f[2]]
        
        pt = closest_point_on_triangle(A, B, C, P)
        dist_sq = np.sum((pt - P)**2)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_pt = pt
            best_face = f_idx
            
    if best_pt is None:
        return mesh.vertices[v], -1
    return best_pt, best_face

# ==========================================
# 3. Voronoi Partitioning & Lloyd Relaxation
# ==========================================

def compute_vertex_adjacency(mesh):
    """Computes a vertex-vertex adjacency list with edge weights."""
    vertices = mesh.vertices
    adjacency = [[] for _ in range(len(vertices))]
    for i, neighbors in enumerate(mesh.vertex_neighbors):
        v_i = vertices[i]
        for j in neighbors:
            v_j = vertices[j]
            weight = np.linalg.norm(v_i - v_j)
            adjacency[i].append((j, weight))
    return adjacency

def select_seeds(mesh, num_seeds):
    """Selects initial seeds on the mesh using Euclidean FPS."""
    vertices = mesh.vertices
    V = len(vertices)
    if num_seeds >= V:
        return list(range(V))
        
    if V > 15000:
        indices = np.random.choice(V, num_seeds, replace=False)
        return list(indices)
        
    seeds = [0]
    min_dists = np.sum((vertices - vertices[0])**2, axis=1)
    
    for _ in range(1, num_seeds):
        next_seed = np.argmax(min_dists)
        seeds.append(next_seed)
        new_dists = np.sum((vertices - vertices[next_seed])**2, axis=1)
        min_dists = np.minimum(min_dists, new_dists)
        
    return seeds

def geodesic_voronoi(mesh, seeds, seed_faces=None, adjacency=None):
    """Partitions mesh vertices into Voronoi cells using continuous seeds."""
    if adjacency is None:
        adjacency = compute_vertex_adjacency(mesh)
        
    V = len(mesh.vertices)
    dist = np.full(V, np.inf)
    nearest_seed = np.full(V, -1, dtype=int)
    
    # Priority Queue: (distance, vertex_idx, seed_idx)
    pq = []
    
    for s_idx, s_pos in enumerate(seeds):
        initialized = False
        if seed_faces is not None and s_idx < len(seed_faces):
            f_idx = seed_faces[s_idx]
            if f_idx != -1:
                # Initialize the three face vertices with their distance to the seed
                f = mesh.faces[f_idx]
                for v in f:
                    d_init = np.linalg.norm(mesh.vertices[v] - s_pos)
                    if d_init < dist[v]:
                        dist[v] = d_init
                        nearest_seed[v] = s_idx
                        heapq.heappush(pq, (d_init, v, s_idx))
                initialized = True
                
        if not initialized:
            sq_dists = np.sum((mesh.vertices - s_pos)**2, axis=1)
            closest_v = np.argmin(sq_dists)
            d_init = np.sqrt(sq_dists[closest_v])
            if d_init < dist[closest_v]:
                dist[closest_v] = d_init
                nearest_seed[closest_v] = s_idx
                heapq.heappush(pq, (d_init, closest_v, s_idx))
                
    while pq:
        d, u, s = heapq.heappop(pq)
        
        if d > dist[u]:
            continue
            
        for v, weight in adjacency[u]:
            nd = d + weight
            if nd < dist[v]:
                dist[v] = nd
                nearest_seed[v] = s
                heapq.heappush(pq, (nd, v, s))
                
    unreachable = (nearest_seed == -1)
    if np.any(unreachable):
        for u in np.where(unreachable)[0]:
            diffs = seeds - mesh.vertices[u]
            sq_dists = np.sum(diffs**2, axis=1)
            closest_idx = np.argmin(sq_dists)
            nearest_seed[u] = closest_idx
            dist[u] = np.sqrt(sq_dists[closest_idx])
            
    return nearest_seed, dist

def lloyd_relaxation(mesh, seeds, num_iterations=3, adjacency=None):
    """
    Applies Lloyd's relaxation on the mesh surface, allowing seeds to float
    freely inside triangles rather than being locked to vertices.
    Creates a 'minimum energy charged particles' configuration.
    """
    if adjacency is None:
        adjacency = compute_vertex_adjacency(mesh)
        
    vertices = mesh.vertices
    curr_seeds = np.array(seeds)
    curr_faces = np.full(len(seeds), -1, dtype=int)
    
    # Set initial seed face indices
    for i, s_pos in enumerate(curr_seeds):
        _, f_idx = project_point_to_mesh(mesh, s_pos)
        curr_faces[i] = f_idx
        
    for _ in range(num_iterations):
        nearest_seed, dist = geodesic_voronoi(mesh, curr_seeds, curr_faces, adjacency)
        new_seeds = []
        new_faces = []
        for s_idx in range(len(curr_seeds)):
            cell_verts = np.where(nearest_seed == s_idx)[0]
            if len(cell_verts) == 0:
                new_seeds.append(curr_seeds[s_idx])
                new_faces.append(curr_faces[s_idx])
                continue
                
            centroid = np.mean(vertices[cell_verts], axis=0)
            proj_pt, f_idx = project_point_to_mesh(mesh, centroid)
            new_seeds.append(proj_pt)
            new_faces.append(f_idx)
            
        curr_seeds = np.array(new_seeds)
        curr_faces = np.array(new_faces)
        
    return curr_seeds, curr_faces

# ==========================================
# 4. Skeleton Extraction & Simplification
# ==========================================

def extract_skeleton(mesh, nearest_seed, dist, tube_boundary_edges=False):
    """
    Extracts the Voronoi skeleton lying on the mesh triangles,
    along with face/vertex normals to define local out-of-plane planes.
    """
    vertices = mesh.vertices
    faces = mesh.faces
    face_normals = mesh.face_normals
    vertex_normals = mesh.vertex_normals
    
    skel_verts = []
    skel_edges = []
    skel_normals = []
    edge_crossover_idx = {}
    
    def get_crossover_idx(u, v, f_idx):
        edge_key = (min(u, v), max(u, v))
        if edge_key in edge_crossover_idx:
            return edge_crossover_idx[edge_key]
            
        t = 0.5
        du = dist[u]
        dv = dist[v]
        edge_len = np.linalg.norm(vertices[u] - vertices[v])
        if edge_len > 1e-8:
            t = 0.5 + (dv - du) / (2.0 * edge_len)
            t = np.clip(t, 0.05, 0.95)
            
        pt = vertices[u] + t * (vertices[v] - vertices[u])
        idx = len(skel_verts)
        skel_verts.append(pt)
        
        # Average normal of edge endpoints
        norm = 0.5 * (vertex_normals[u] + vertex_normals[v])
        n_len = np.linalg.norm(norm)
        if n_len > 1e-6:
            norm = norm / n_len
        else:
            norm = face_normals[f_idx]
        skel_normals.append(norm)
        
        edge_crossover_idx[edge_key] = idx
        return idx

    # Run over all triangles to find cell boundaries
    for f_idx, f in enumerate(faces):
        c1, c2, c3 = nearest_seed[f[0]], nearest_seed[f[1]], nearest_seed[f[2]]
        unique_seeds = len(set([c1, c2, c3]))
        
        if unique_seeds == 1:
            continue
        elif unique_seeds == 2:
            v1, v2, v3 = f[0], f[1], f[2]
            c1, c2, c3 = nearest_seed[v1], nearest_seed[v2], nearest_seed[v3]
            
            if c1 == c2:
                idx_a = get_crossover_idx(v1, v3, f_idx)
                idx_b = get_crossover_idx(v2, v3, f_idx)
            elif c2 == c3:
                idx_a = get_crossover_idx(v1, v2, f_idx)
                idx_b = get_crossover_idx(v1, v3, f_idx)
            else:
                idx_a = get_crossover_idx(v1, v2, f_idx)
                idx_b = get_crossover_idx(v2, v3, f_idx)
                
            skel_edges.append((idx_a, idx_b))
            
        elif unique_seeds == 3:
            v1, v2, v3 = f[0], f[1], f[2]
            pt_junction = (vertices[v1] + vertices[v2] + vertices[v3]) / 3.0
            idx_junction = len(skel_verts)
            skel_verts.append(pt_junction)
            skel_normals.append(face_normals[f_idx])
            
            idx_12 = get_crossover_idx(v1, v2, f_idx)
            idx_23 = get_crossover_idx(v2, v3, f_idx)
            idx_31 = get_crossover_idx(v3, v1, f_idx)
            
            skel_edges.append((idx_junction, idx_12))
            skel_edges.append((idx_junction, idx_23))
            skel_edges.append((idx_junction, idx_31))

    # Add open border edges if checked
    if tube_boundary_edges:
        counts = np.bincount(mesh.edges_unique_inverse)
        boundary_edges = mesh.edges_unique[counts == 1]
        
        if len(boundary_edges) > 0:
            orig_to_skel_boundary = {}
            for u, v in boundary_edges:
                for w in (u, v):
                    if w not in orig_to_skel_boundary:
                        idx = len(skel_verts)
                        skel_verts.append(vertices[w])
                        skel_normals.append(vertex_normals[w])
                        orig_to_skel_boundary[w] = idx
                skel_edges.append((orig_to_skel_boundary[u], orig_to_skel_boundary[v]))

    return np.array(skel_verts), skel_edges, np.array(skel_normals)

def simplify_skeleton(skel_verts, skel_edges, skel_normals):
    """
    Simplifies the skeleton by bypassing intermediate degree-2 crossover points
    and connecting junction nodes and boundary endpoints with straight segments.
    """
    if len(skel_verts) == 0 or len(skel_edges) == 0:
        return skel_verts, skel_edges, skel_normals
        
    V = len(skel_verts)
    adj = [set() for _ in range(V)]
    for u, v in skel_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find connected components using BFS
    visited = np.zeros(V, dtype=bool)
    components = []
    for i in range(V):
        if not visited[i] and len(adj[i]) > 0:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for n in adj[curr]:
                    if not visited[n]:
                        visited[n] = True
                        queue.append(n)
            components.append(comp)
            
    simplified_verts = []
    simplified_edges = []
    simplified_normals = []
    orig_to_sim = {}
    
    def get_sim_idx(idx):
        if idx not in orig_to_sim:
            sim_idx = len(simplified_verts)
            simplified_verts.append(skel_verts[idx])
            simplified_normals.append(skel_normals[idx])
            orig_to_sim[idx] = sim_idx
        return orig_to_sim[idx]
        
    for comp in components:
        comp_key_nodes = [node for node in comp if len(adj[node]) != 2]
        
        if len(comp_key_nodes) == 0:
            comp_key_nodes = [comp[0]]
            
        traced_edges = set()
        for start_node in comp_key_nodes:
            for neighbor in adj[start_node]:
                edge_key = (min(start_node, neighbor), max(start_node, neighbor))
                if edge_key in traced_edges:
                    continue
                traced_edges.add(edge_key)
                
                prev = start_node
                curr = neighbor
                path = [start_node, curr]
                
                while len(adj[curr]) == 2 and curr not in comp_key_nodes:
                    neighbors = list(adj[curr])
                    next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
                    prev = curr
                    curr = next_node
                    path.append(curr)
                    
                    step_key = (min(prev, curr), max(prev, curr))
                    traced_edges.add(step_key)
                    
                if start_node == curr:
                    prev_idx = get_sim_idx(start_node)
                    for pt_idx in path[1:-1]:
                        curr_idx = get_sim_idx(pt_idx)
                        simplified_edges.append((prev_idx, curr_idx))
                        prev_idx = curr_idx
                    simplified_edges.append((prev_idx, get_sim_idx(curr)))
                else:
                    idx_start = get_sim_idx(start_node)
                    idx_end = get_sim_idx(curr)
                    simplified_edges.append((idx_start, idx_end))
                    
    return np.array(simplified_verts), simplified_edges, np.array(simplified_normals)

# ==========================================
# 5. Voxelization and Marching Cubes
# ==========================================

def generate_tube_mesh(skel_verts, skel_edges, skel_normals, tube_radius, k_blend, grid_res, pad=0.15):
    """
    Computes a localized Signed Distance Field (SDF) of the skeleton network
    and extracts a watertight tube mesh using Marching Cubes.
    
    Uses a 'flat pill' formulation:
    - The smooth minimum is isolated to the in-plane components of the tube distances
      to create perfectly smooth, wobbly-free parabolic fillets (like a string art curve).
    - The out-of-plane component is processed separately, ensuring that the height
      perpendicular to the local surface remains strictly capped at the tube radius,
      completely eliminating any out-of-plane bulging at the nodes.
    """
    if len(skel_verts) == 0 or len(skel_edges) == 0:
        return None
        
    skel_min = np.min(skel_verts, axis=0)
    skel_max = np.max(skel_verts, axis=0)
    
    diag = np.linalg.norm(skel_max - skel_min)
    padding = max(pad * diag, 3.0 * (tube_radius + k_blend))
    min_coords = skel_min - padding
    max_coords = skel_max + padding
    
    voxel_size = (max_coords - min_coords) / (grid_res - 1)
    
    # Initialize voxel grids
    sdf_in_plane = np.full((grid_res, grid_res, grid_res), 1e9, dtype=np.float32)
    sdf_h = np.zeros((grid_res, grid_res, grid_res), dtype=np.float32)
    min_d_in_plane = np.full((grid_res, grid_res, grid_res), 1e9, dtype=np.float32)
    
    x = np.linspace(min_coords[0], max_coords[0], grid_res, dtype=np.float32)
    y = np.linspace(min_coords[1], max_coords[1], grid_res, dtype=np.float32)
    z = np.linspace(min_coords[2], max_coords[2], grid_res, dtype=np.float32)
    
    def smin_poly(a, b, k):
        if k <= 0:
            return np.minimum(a, b)
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return np.minimum(a, b) - k * h * (1.0 - h)
        
    # Process Edges (Tubes)
    for u, v in skel_edges:
        p_A = skel_verts[u]
        p_B = skel_verts[v]
        
        n_A = skel_normals[u]
        n_B = skel_normals[v]
        n_seg = 0.5 * (n_A + n_B)
        n_len = np.linalg.norm(n_seg)
        if n_len > 1e-6:
            n_seg = n_seg / n_len
        else:
            n_seg = np.array([0.0, 0.0, 1.0])
            
        crop_margin = tube_radius + k_blend + 2.0 * np.max(voxel_size)
        seg_min = np.minimum(p_A, p_B) - crop_margin
        seg_max = np.maximum(p_A, p_B) + crop_margin
        
        i_min = max(0, int(np.floor((seg_min[0] - min_coords[0]) / voxel_size[0])))
        i_max = min(grid_res - 1, int(np.ceil((seg_max[0] - min_coords[0]) / voxel_size[0])))
        j_min = max(0, int(np.floor((seg_min[1] - min_coords[1]) / voxel_size[1])))
        j_max = min(grid_res - 1, int(np.ceil((seg_max[1] - min_coords[1]) / voxel_size[1])))
        k_min = max(0, int(np.floor((seg_min[2] - min_coords[2]) / voxel_size[2])))
        k_max = min(grid_res - 1, int(np.ceil((seg_max[2] - min_coords[2]) / voxel_size[2])))
        
        if i_min > i_max or j_min > j_max or k_min > k_max:
            continue
            
        grid_x, grid_y, grid_z = np.meshgrid(
            x[i_min:i_max+1],
            y[j_min:j_max+1],
            z[k_min:k_max+1],
            indexing='ij'
        )
        
        pts = np.stack([grid_x, grid_y, grid_z], axis=-1)
        v_vec = pts - p_A
        
        # Out-of-plane component along normal
        h = np.sum(v_vec * n_seg, axis=-1)
        
        # In-plane component
        v_in_plane = v_vec - h[..., np.newaxis] * n_seg
        
        # In-plane distance calculation
        AB = p_B - p_A
        AB_len_sq = np.dot(AB, AB)
        if AB_len_sq < 1e-8:
            d_in_plane = np.linalg.norm(v_in_plane, axis=-1)
        else:
            t = np.sum(v_in_plane * AB, axis=-1) / AB_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj = t[..., np.newaxis] * AB
            d_in_plane = np.linalg.norm(v_in_plane - proj, axis=-1)
            
        # 1. Smooth min of in-plane distance (shapes the parabolic chamfer fillet)
        sdf_in_plane[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = smin_poly(
            sdf_in_plane[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1],
            d_in_plane,
            k_blend
        )
        
        # 2. Track height relative to closest segment to apply flat capping
        mask = d_in_plane < min_d_in_plane[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1]
        min_d_in_plane[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1][mask] = d_in_plane[mask]
        sdf_h[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1][mask] = h[mask]
        
    # Combine in-plane and normal components quadratically.
    # Capped strictly out-of-plane, round cross section on straight parts.
    sdf_3d = np.sqrt(np.maximum(sdf_in_plane, 0.0)**2 + sdf_h**2) - tube_radius
    
    try:
        verts, faces, normals, values = marching_cubes(sdf_3d, level=0.0)
        verts_world = min_coords + verts * voxel_size
        return trimesh.Trimesh(vertices=verts_world, faces=faces)
    except Exception as e:
        print(f"Error in marching_cubes: {e}")
        return None

# ==========================================
# 6. Visualization Helpers
# ==========================================

def compute_shaded_colors(tri_mesh, base_color=(0.95, 0.6, 0.2), light_dir=None):
    """Calculates flat-shaded lighting intensities for rendering."""
    if light_dir is None:
        light_dir = np.array([1.0, 1.0, 2.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    normals = tri_mesh.face_normals
    cos_angle = np.dot(normals, light_dir)
    intensity = 0.3 + 0.7 * np.clip((cos_angle + 1) / 2.0, 0.0, 1.0)
    
    colors = np.outer(intensity, base_color)
    return colors

# ==========================================
# 7. Tkinter GUI Application
# ==========================================

class OrganicMeshApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Organic Mesh Tube Structure Generator")
        self.geometry("1100x780")
        self.configure(bg='#1e1e1e')
        
        self.workspace_dir = os.path.dirname(os.path.abspath(__file__))
        
        # State
        self.input_mesh = None
        self.mesh_scale = 1.0
        self.seeds_3d = None
        self.seed_faces = None
        self.nearest_seed = None
        self.geodesic_dist = None
        self.adjacency = None
        self.skel_verts = None
        self.skel_edges = None
        self.skel_normals = None
        self.tube_mesh = None
        self.file_name = "None"
        self.cached_lloyd_steps = -1
        
        # Style Configuration
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#e1e1e1')
        
        self.setup_layout()
        
    def setup_layout(self):
        # 1. Left Control Panel
        sidebar = tk.Frame(self, bg='#252526', width=320)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # Title
        title = tk.Label(sidebar, text="Organic Mesh Generator", fg='#ffffff', bg='#252526', font=('Arial', 14, 'bold'))
        title.pack(pady=15, padx=10, anchor='w')
        
        # Loader Frame
        loader_frame = tk.LabelFrame(sidebar, text="Mesh Selection", fg='#e1e1e1', bg='#252526', bd=1, relief=tk.FLAT, font=('Arial', 10, 'bold'))
        loader_frame.pack(fill=tk.X, padx=12, pady=5)
        
        load_btn = tk.Button(loader_frame, text="Load Custom .STL File", command=self.load_file, bg='#007acc', fg='#ffffff', activebackground='#0062a3', bd=0, padx=10, pady=6, font=('Arial', 10, 'bold'))
        load_btn.pack(fill=tk.X, padx=10, pady=6)
        
        self.file_label = tk.Label(loader_frame, text="File: None", fg='#aaaaaa', bg='#252526', anchor='w', font=('Arial', 9))
        self.file_label.pack(fill=tk.X, padx=10, pady=2)
        
        test_frame = tk.Frame(loader_frame, bg='#252526')
        test_frame.pack(fill=tk.X, padx=10, pady=6)
        
        sphere_btn = tk.Button(test_frame, text="Sphere Test", command=self.load_test_sphere, bg='#3e3e3f', fg='#ffffff', activebackground='#505051', bd=0, padx=8, pady=4, font=('Arial', 9))
        sphere_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,4))
        
        cylinder_btn = tk.Button(test_frame, text="Open Cylinder", command=self.load_test_cylinder, bg='#3e3e3f', fg='#ffffff', activebackground='#505051', bd=0, padx=8, pady=4, font=('Arial', 9))
        cylinder_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(4,0))
        
        # Parameters Frame
        params_frame = tk.LabelFrame(sidebar, text="Parameters", fg='#e1e1e1', bg='#252526', bd=1, relief=tk.FLAT, font=('Arial', 10, 'bold'))
        params_frame.pack(fill=tk.X, padx=12, pady=5)
        
        # Sliders
        self.create_slider(params_frame, "Voronoi Cell Density", "cell_count_var", 10, 200, 40, 5, self.update_cell_label)
        self.create_slider(params_frame, "Lloyd Relaxation Steps", "lloyd_steps_var", 0, 15, 5, 1, self.update_lloyd_label)
        self.create_slider(params_frame, "Tube Thickness (% scale)", "tube_radius_var", 0.1, 5.0, 1.0, 0.1, self.update_radius_label)
        self.create_slider(params_frame, "Junction Transition (Fillet Size)", "fillet_mult_var", 0.0, 4.0, 1.0, 0.1, self.update_fillet_label)
        self.create_slider(params_frame, "Voxel Grid Resolution", "grid_res_var", 32, 128, 104, 4, self.update_res_label)
        
        # Options Frame
        opts_frame = tk.LabelFrame(sidebar, text="Options", fg='#e1e1e1', bg='#252526', bd=1, relief=tk.FLAT, font=('Arial', 10, 'bold'))
        opts_frame.pack(fill=tk.X, padx=12, pady=5)
        
        self.skeleton_only_var = tk.BooleanVar(value=False)
        skel_chk = tk.Checkbutton(opts_frame, text="Skeleton View (Fast)", variable=self.skeleton_only_var, command=self.regenerate, bg='#252526', fg='#e1e1e1', selectcolor='#1e1e1e', activebackground='#252526', activeforeground='#e1e1e1')
        skel_chk.pack(anchor='w', padx=10, pady=3)
        
        self.show_centers_var = tk.BooleanVar(value=True)
        centers_chk = tk.Checkbutton(opts_frame, text="Show Cell Centers", variable=self.show_centers_var, command=self.update_plot, bg='#252526', fg='#e1e1e1', selectcolor='#1e1e1e', activebackground='#252526', activeforeground='#e1e1e1')
        centers_chk.pack(anchor='w', padx=10, pady=3)
        
        self.tube_boundaries_var = tk.BooleanVar(value=True)
        bound_chk = tk.Checkbutton(opts_frame, text="Tube open boundary edges", variable=self.tube_boundaries_var, command=self.force_regenerate, bg='#252526', fg='#e1e1e1', selectcolor='#1e1e1e', activebackground='#252526', activeforeground='#e1e1e1')
        bound_chk.pack(anchor='w', padx=10, pady=3)
        
        # Actions Frame
        action_frame = tk.Frame(sidebar, bg='#252526')
        action_frame.pack(fill=tk.X, padx=12, pady=8)
        
        regen_btn = tk.Button(action_frame, text="Regenerate", command=self.force_regenerate, bg='#3e3e3f', fg='#ffffff', activebackground='#505051', bd=0, padx=10, pady=6, font=('Arial', 10, 'bold'))
        regen_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        
        self.export_btn = tk.Button(action_frame, text="Export STL...", command=self.export_stl, bg='#007acc', fg='#ffffff', activebackground='#0062a3', bd=0, padx=10, pady=6, state=tk.DISABLED, font=('Arial', 10, 'bold'))
        self.export_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(4, 0))
        
        # Status
        status_box = tk.Frame(sidebar, bg='#1e1e24', height=40)
        status_box.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(status_box, text="Status: Ready. Please load a mesh.", fg='#aaaaaa', bg='#1e1e24', font=('Arial', 9), anchor='w', wraplength=280, justify='left')
        self.status_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 2. Right Canvas Area
        self.plot_frame = tk.Frame(self, bg='#121212')
        self.plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.fig = plt.figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_axis_off()
        self.ax.set_facecolor('#121212')
        self.fig.patch.set_facecolor('#121212')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_slider(self, parent, label_text, var_name, from_val, to_val, default_val, resolution, update_fn):
        frame = tk.Frame(parent, bg='#252526')
        frame.pack(fill=tk.X, pady=2, padx=10)
        
        label = tk.Label(frame, text=label_text, fg='#e1e1e1', bg='#252526', font=('Arial', 9, 'bold'), anchor='w')
        label.pack(fill=tk.X)
        
        val_label = tk.Label(frame, text="", fg='#aaaaaa', bg='#252526', font=('Arial', 8), anchor='w')
        val_label.pack(fill=tk.X)
        
        var = tk.DoubleVar(value=default_val)
        setattr(self, var_name, var)
        
        def on_change(val):
            val_float = float(val)
            update_fn(val_float, val_label)
            
        scale = tk.Scale(
            frame,
            from_=from_val,
            to=to_val,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=var,
            showvalue=False,
            command=on_change,
            bg='#252526',
            fg='#e1e1e1',
            troughcolor='#1e1e1e',
            activebackground='#007acc',
            highlightthickness=0,
            bd=0
        )
        scale.pack(fill=tk.X, pady=1)
        scale.bind("<ButtonRelease-1>", lambda event: self.regenerate())
        
        setattr(self, var_name + "_label_fn", lambda: on_change(var.get()))
        on_change(default_val)
        
    # Label updates
    def update_cell_label(self, val, label):
        label.config(text=f"{int(val)} partition regions")
        
    def update_lloyd_label(self, val, label):
        label.config(text=f"{int(val)} steps (smooth relaxation inside triangles)")
        
    def update_radius_label(self, val, label):
        phys = val * 0.01 * self.mesh_scale
        label.config(text=f"{val:.2f}% of mesh diagonal ({phys:.4f} units)")
        
    def update_fillet_label(self, val, label):
        t_rad = self.tube_radius_var.get() * 0.01 * self.mesh_scale if hasattr(self, 'tube_radius_var') else 0.01
        phys = val * t_rad
        label.config(text=f"{val:.2f}x tube radius ({phys:.4f} units - concave fillet chamfer)")
        
    def update_res_label(self, val, label):
        label.config(text=f"{int(val)}^3 voxels")
        
    def refresh_labels(self):
        for var_name in ["cell_count_var", "lloyd_steps_var", "tube_radius_var", "fillet_mult_var", "grid_res_var"]:
            if hasattr(self, var_name + "_label_fn"):
                getattr(self, var_name + "_label_fn")()
                
    # Loading functions
    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("STL files", "*.stl")])
        if path:
            self.load_stl_from_path(path)
            
    def load_test_sphere(self):
        path = os.path.join(self.workspace_dir, "test_sphere.stl")
        self.status_label.config(text="Generating sphere STL test case...")
        self.update_idletasks()
        
        mesh = make_test_sphere(radius=1.0, subdivisions=3)
        mesh.export(path)
        self.load_stl_from_path(path)
        
    def load_test_cylinder(self):
        path = os.path.join(self.workspace_dir, "test_cylinder.stl")
        self.status_label.config(text="Generating open cylinder STL test case...")
        self.update_idletasks()
        
        mesh = make_open_cylinder(radius=1.0, height=2.0, sections=24, segments=12)
        mesh.export(path)
        self.load_stl_from_path(path)
        
    def load_stl_from_path(self, path):
        try:
            self.status_label.config(text="Loading STL...")
            self.update_idletasks()
            
            mesh = trimesh.load(path)
            mesh.merge_vertices()
            
            if len(mesh.vertices) == 0:
                raise ValueError("Mesh has no vertices.")
                
            self.input_mesh = mesh
            self.file_name = os.path.basename(path)
            self.file_label.config(text=f"File: {self.file_name} ({len(self.input_mesh.faces)} faces)")
            
            bbox = self.input_mesh.bounds
            self.mesh_scale = np.linalg.norm(bbox[1] - bbox[0])
            if self.mesh_scale < 1e-8:
                self.mesh_scale = 1.0
                
            self.status_label.config(text="Preparing mesh connectivity topology...")
            self.update_idletasks()
            
            self.adjacency = compute_vertex_adjacency(self.input_mesh)
            self.seeds_3d = None
            self.seed_faces = None
            self.cached_lloyd_steps = -1
            
            self.refresh_labels()
            self.regenerate()
            
        except Exception as e:
            messagebox.showerror("Error Loading Mesh", f"Could not load the file:\n{str(e)}")
            self.status_label.config(text="Status: Error loading mesh.")
            
    def force_regenerate(self):
        self.seeds_3d = None
        self.regenerate()
        
    def regenerate(self):
        if self.input_mesh is None:
            self.status_label.config(text="Status: No mesh loaded.")
            return
            
        self.status_label.config(text="Status: Partitioning surfaces...")
        self.update_idletasks()
        
        t0 = time.time()
        
        num_seeds = int(self.cell_count_var.get())
        lloyd_steps = int(self.lloyd_steps_var.get())
        tube_rad_pct = self.tube_radius_var.get()
        fillet_mult = self.fillet_mult_var.get()
        grid_res = int(self.grid_res_var.get())
        self.skeleton_only = self.skeleton_only_var.get()
        tube_boundaries = self.tube_boundaries_var.get()
        
        tube_radius = tube_rad_pct * 0.01 * self.mesh_scale
        k_blend = fillet_mult * tube_radius
        
        # Run seed selection & Lloyd relaxation if cell parameters changed
        if (self.seeds_3d is None or 
            len(self.seeds_3d) != num_seeds or 
            self.cached_lloyd_steps != lloyd_steps):
            
            init_indices = select_seeds(self.input_mesh, num_seeds)
            self.seeds_3d = self.input_mesh.vertices[init_indices]
            self.seed_faces = np.full(num_seeds, -1, dtype=int)
            
            if lloyd_steps > 0:
                self.status_label.config(text="Status: Relaxing cell centers (Lloyd CVT)...")
                self.update_idletasks()
                self.seeds_3d, self.seed_faces = lloyd_relaxation(
                    self.input_mesh, self.seeds_3d, lloyd_steps, self.adjacency
                )
                
            self.nearest_seed, self.geodesic_dist = geodesic_voronoi(
                self.input_mesh, self.seeds_3d, self.seed_faces, self.adjacency
            )
            self.cached_lloyd_steps = lloyd_steps
            
        raw_verts, raw_edges, raw_normals = extract_skeleton(
            self.input_mesh, self.nearest_seed, self.geodesic_dist, tube_boundaries
        )
        self.skel_verts, self.skel_edges, self.skel_normals = simplify_skeleton(raw_verts, raw_edges, raw_normals)
        
        t_skel = time.time() - t0
        
        if self.skeleton_only:
            self.tube_mesh = None
            self.status_label.config(text=f"Status: Skeleton built in {t_skel:.3f}s. (Nodes: {len(self.skel_verts)}, Edges: {len(self.skel_edges)})")
            self.export_btn.config(state=tk.DISABLED)
        else:
            self.status_label.config(text="Status: Generating organic tubes (voxelizing)...")
            self.update_idletasks()
            
            t_vox_start = time.time()
            self.tube_mesh = generate_tube_mesh(
                self.skel_verts, self.skel_edges, self.skel_normals, tube_radius, k_blend, grid_res
            )
            t_vox = time.time() - t_vox_start
            total = time.time() - t0
            
            if self.tube_mesh is not None:
                self.status_label.config(
                    text=f"Status: Watertight tubes created in {total:.3f}s (Skel: {t_skel:.2f}s, Voxel: {t_vox:.2f}s) | Faces: {len(self.tube_mesh.faces)}"
                )
                self.export_btn.config(state=tk.NORMAL)
            else:
                self.status_label.config(text="Status: Voxelization / Marching Cubes failed.")
                self.export_btn.config(state=tk.DISABLED)
                
        self.update_plot()
        
    def update_plot(self):
        self.ax.clear()
        self.ax.set_axis_off()
        
        self.ax.set_facecolor('#121212')
        self.fig.patch.set_facecolor('#121212')
        
        if self.input_mesh is not None:
            poly_input = Poly3DCollection(self.input_mesh.triangles, alpha=0.07, facecolors='#888888', edgecolors='#555555', linewidths=0.1)
            self.ax.add_collection3d(poly_input)
            
        if self.skeleton_only:
            if self.skel_verts is not None and len(self.skel_verts) > 0 and len(self.skel_edges) > 0:
                segments = [ [self.skel_verts[u], self.skel_verts[v]] for u, v in self.skel_edges ]
                line_coll = Line3DCollection(segments, colors='#00bcff', linewidths=1.5, alpha=0.8)
                self.ax.add_collection3d(line_coll)
        else:
            if self.tube_mesh is not None and len(self.tube_mesh.faces) > 0:
                shaded_colors = compute_shaded_colors(self.tube_mesh, base_color=(0.95, 0.6, 0.2))
                poly_tubes = Poly3DCollection(self.tube_mesh.triangles, facecolors=shaded_colors, edgecolors='none', alpha=1.0)
                self.ax.add_collection3d(poly_tubes)
                
        # Draw Cell Centers if checked
        if self.input_mesh is not None and self.show_centers_var.get():
            if self.seeds_3d is not None and len(self.seeds_3d) > 0:
                self.ax.scatter(self.seeds_3d[:, 0], self.seeds_3d[:, 1], self.seeds_3d[:, 2],
                                c='#a3ff00', s=25, depthshade=True, edgecolors='#121212', linewidths=0.5, label='Cell Centers')
                
        if self.input_mesh is not None:
            bounds = self.input_mesh.bounds
            center = np.mean(bounds, axis=0)
            extents = bounds[1] - bounds[0]
            max_extent = np.max(extents)
            half_extent = 0.55 * max_extent
            
            self.ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
            self.ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
            self.ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
            self.ax.set_box_aspect((1, 1, 1))
            
        self.canvas.draw()
        
    def export_stl(self):
        if self.tube_mesh is None:
            messagebox.showwarning("No mesh", "No tube mesh has been generated to export.")
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])
        if path:
            try:
                self.status_label.config(text="Exporting STL...")
                self.update_idletasks()
                
                self.tube_mesh.export(path)
                self.status_label.config(text=f"Status: Successfully exported to {os.path.basename(path)}")
                messagebox.showinfo("Export Successful", f"Mesh exported successfully to:\n{path}")
                
            except Exception as e:
                messagebox.showerror("Export Failed", f"Could not export STL file:\n{str(e)}")
                self.status_label.config(text="Status: STL export failed.")

if __name__ == '__main__':
    app = OrganicMeshApp()
    app.mainloop()
