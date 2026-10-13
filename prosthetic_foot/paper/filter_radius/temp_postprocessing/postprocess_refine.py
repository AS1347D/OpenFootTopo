# adios_read.py
from dolfinx.fem import (Function, functionspace, create_interpolation_data)
from dolfinx.mesh import compute_incident_entities, refine
from dolfinx import plot
from mpi4py import MPI
import adios4dolfinx
import dolfinx
from pathlib import Path
import basix
import numpy as np

path_out = Path("adios_dir/checkpoint.bp")
path_pyvista = Path("adios_dir/laplacian.png")
path_pyvista_refined = Path("adios_dir/refined.png")
path_stl = Path("printing/preparedGeometry.obj")

domain = adios4dolfinx.read_mesh(path_out, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

S = functionspace(domain, ("CG", 1))

rho_import = Function(S)

adios4dolfinx.read_function(path_out, rho_import, engine="BP4")



from topolegy.io_utils import pyvista_visualizer, xdmf_writer

# pyvista_visualizer(path_pyvista, rho_import, show_mesh=False)

# dg0_el = basix.ufl.element("DG", domain.basix_cell(), 0)
# Q = functionspace(domain, dg0_el)
# rho0 = Function(Q)
# rho0.interpolate(rho_import)  # cellwise constant approximation


path_refined = Path("adios_dir/mesh_smoothing.bp")

domain_smoohing = adios4dolfinx.read_mesh(path_refined, MPI.COMM_WORLD, engine='BP4', ghost_mode=dolfinx.mesh.GhostMode.shared_facet)

S_refined = functionspace(domain_smoohing, ("CG", 1))

rho_refined = Function(S_refined)

padding = 1e-14

fine_mesh_cell_map = domain_smoohing.topology.index_map(domain_smoohing.topology.dim)
num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
cells = np.arange(num_cells_on_proc, dtype=np.int32)
interpolation_data = create_interpolation_data(S_refined, S, cells, padding=padding)

rho_refined.interpolate_nonmatching(rho_import, cells, interpolation_data=interpolation_data)


xdmf_writer("printing/refined", domain_smoohing, rho_refined)

# postprocess_rbf_rdp_holes.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpi4py import MPI

import dolfinx
from dolfinx import plot
from dolfinx.fem import Function, functionspace
import adios4dolfinx

from scipy.interpolate import splprep, splev, RBFInterpolator
from scipy.spatial import cKDTree

try:
    from rdp import rdp
except ImportError as e:
    raise ImportError("Please install rdp:  pip install rdp") from e


# ------------------------- helpers -------------------------

def get_ordered_polyline_points(poly: pv.PolyData, close: bool = True):
    """
    Return ordered point coordinates for each connected polyline in a PyVista PolyData.
    Sorted by number of points (largest first). Works on older PyVista.
    """
    poly = poly.clean()
    if poly.lines is None or poly.lines.size == 0:
        raise ValueError("PolyData has no line cells. Pass extract_feature_edges output.")

    lines_arr = np.asarray(poly.lines, dtype=np.int64)
    pts = np.asarray(poly.points)

    # Build undirected adjacency from line cells
    edges = []
    i = 0
    while i < len(lines_arr):
        n = int(lines_arr[i]); i += 1
        ids = lines_arr[i:i+n]; i += n
        for a, b in zip(ids[:-1], ids[1:]):
            a, b = int(a), int(b)
            if a != b:
                edges.append((a, b))
    if not edges:
        raise ValueError("No edges parsed from poly.lines.")

    from collections import defaultdict, deque
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)

    # Connected components over vertices
    nodes = set(adj.keys())
    seen = set()
    comps = []
    for s in list(nodes):
        if s in seen:
            continue
        q = deque([s]); seen.add(s)
        comp = []
        while q:
            v = q.popleft()
            comp.append(v)
            for nb in adj[v]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)

    # Walk each component into an ordered polyline
    out = []
    for comp in comps:
        comp = set(comp)
        deg = {v: sum(1 for nb in adj[v] if nb in comp) for v in comp}
        endpoints = [v for v, d in deg.items() if d == 1]

        start = endpoints[0] if endpoints else min(comp)

        order = [start]
        prev = None
        cur = start
        while True:
            nbrs = [nb for nb in adj[cur] if nb in comp and nb != prev]
            if not nbrs:
                break
            nxt = nbrs[0] if len(nbrs) == 1 else min(nbrs)

            if nxt == start:
                order.append(nxt)
                break
            order.append(nxt)
            prev, cur = cur, nxt

            if len(order) > len(comp) + 5:
                break

        coords = pts[np.array(order, dtype=np.int64)]
        if close and np.linalg.norm(coords[0] - coords[-1]) > 1e-12 and not endpoints:
            coords = np.vstack([coords, coords[0]])
        out.append(coords)

    out.sort(key=lambda a: a.shape[0], reverse=True)
    return out


def ensure_closed(pts2: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts2, float)
    if pts.shape[0] < 3:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[0]])
    return pts


def rdp_simplify_closed(pts2: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    RDP simplification for a closed polyline. eps in same units as coords (mm).
    """
    pts = ensure_closed(pts2)
    # drop last repeated point for rdp
    pts_open = pts[:-1]
    simp = np.asarray(rdp(pts_open, epsilon=eps), float)
    simp = ensure_closed(simp)
    return simp


def resample_by_arclength_closed(pts2: np.ndarray, n: int = 200) -> np.ndarray:
    """
    Resample a closed polyline (Nx2) to n evenly spaced points (returns Nx2, not repeated end).
    """
    pts = ensure_closed(pts2)
    # cumulative arclength
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    if L <= 0:
        return pts[:-1].copy()

    s_new = np.linspace(0.0, L, n+1)[:-1]
    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])
    return np.c_[x_new, y_new]


def smooth_closed_spline(ctrl: np.ndarray, n_out: int = 800, s: float = 25.0) -> np.ndarray:
    """Periodic B-spline smoothing on closed curve control points (Nx2)."""
    x, y = ctrl[:, 0], ctrl[:, 1]
    tck, _ = splprep([x, y], s=s, per=True)
    u = np.linspace(0.0, 1.0, n_out, endpoint=False)
    xs, ys = splev(u, tck)
    return np.c_[xs, ys]


def map_to_nearest(src: np.ndarray, target_curve: np.ndarray) -> np.ndarray:
    tree = cKDTree(target_curve)
    _, idx = tree.query(src)
    return target_curve[idx]


def rbf_deform_points(pts_all: np.ndarray, ctrl_src: np.ndarray, ctrl_tgt: np.ndarray,
                      smoothing: float = 1e-6, add_anchors: bool = True) -> np.ndarray:
    Xc = np.asarray(ctrl_src, float)
    Dc = np.asarray(ctrl_tgt, float) - Xc

    if add_anchors:
        bb_min = pts_all.min(axis=0)
        bb_max = pts_all.max(axis=0)
        pad = 0.25 * (bb_max - bb_min)
        anchors = np.array([
            bb_min - pad,
            [bb_max[0] + pad[0], bb_min[1] - pad[1]],
            bb_max + pad,
            [bb_min[0] - pad[0], bb_max[1] + pad[1]],
        ])
        Xc = np.vstack([Xc, anchors])
        Dc = np.vstack([Dc, np.zeros((anchors.shape[0], 2))])

    rbf = RBFInterpolator(Xc, Dc, kernel="thin_plate_spline", smoothing=smoothing)
    return pts_all + rbf(pts_all)


# Build PyVista grid (use mesh)
tdim = domain_smoohing.topology.dim
topo, cell_types, geom = plot.vtk_mesh(domain_smoohing, tdim)
u_grid = pv.UnstructuredGrid(topo, cell_types, geom)
u_grid.point_data["u"] = rho_refined.x.array.real

# Threshold and extract boundary edges (outer + holes)
rho_thr = 0.6
solid = u_grid.threshold(value=rho_thr, scalars="u")
edges = solid.extract_feature_edges(
    boundary_edges=True, feature_edges=False,
    manifold_edges=False, non_manifold_edges=False
).clean()

loops3 = get_ordered_polyline_points(edges, close=True)
if len(loops3) == 0:
    raise RuntimeError("No boundary loops found. Try a different threshold.")

outer_loop = loops3[0][:, :2]              # largest loop
hole_loops = [ln[:, :2] for ln in loops3[1:]]

print(f"Found loops: outer(pts={outer_loop.shape[0]}), holes={len(hole_loops)}")

# ---- parameters you tune ----
rdp_eps_outer = 1.0   # mm
rdp_eps_hole  = 0.8   # mm
n_ctrl_outer  = 300
n_ctrl_hole   = 160
s_spline_outer = 30.0
s_spline_hole  = 20.0
rbf_smoothing  = 1e-6
# -----------------------------

def build_controls(loop_pts, rdp_eps, n_ctrl, s_spline, n_out):
    """loop -> simplified ctrl -> spline smooth -> target ctrl by nearest"""
    simp = rdp_simplify_closed(loop_pts, eps=rdp_eps)
    ctrl = resample_by_arclength_closed(simp, n=n_ctrl)
    smooth_curve = smooth_closed_spline(ctrl, n_out=n_out, s=s_spline)
    target = map_to_nearest(ctrl, smooth_curve)
    return ctrl, target, smooth_curve

# Controls for outer
outer_ctrl, outer_target, outer_smooth = build_controls(
    outer_loop, rdp_eps_outer, n_ctrl_outer, s_spline_outer, n_out=900
)

# Controls for holes (included!)
hole_ctrls = []
hole_targets = []
hole_smooths = []
for h in hole_loops:
    hc, ht, hs = build_controls(h, rdp_eps_hole, n_ctrl_hole, s_spline_hole, n_out=500)
    hole_ctrls.append(hc)
    hole_targets.append(ht)
    hole_smooths.append(hs)

# Stack all control constraints: outer + holes
ctrl_src = np.vstack([outer_ctrl] + hole_ctrls) if hole_ctrls else outer_ctrl
ctrl_tgt = np.vstack([outer_target] + hole_targets) if hole_targets else outer_target

# Deform all "solid" points (2D)
P_all = solid.points[:, :2]
P_def = rbf_deform_points(P_all, ctrl_src, ctrl_tgt, smoothing=rbf_smoothing, add_anchors=True)

solid_def = solid.copy()
solid_def.points[:, 0] = P_def[:, 0]
solid_def.points[:, 1] = P_def[:, 1]

# Extract deformed edges and re-get loops for plotting
edges_def = solid_def.extract_feature_edges(
    boundary_edges=True, feature_edges=False,
    manifold_edges=False, non_manifold_edges=False
).clean()
loops_def3 = get_ordered_polyline_points(edges_def, close=True)
outer_def = loops_def3[0][:, :2]
holes_def = [ln[:, :2] for ln in loops_def3[1:]]

# ---- plot: raw vs smoothed (outer + holes) ----
plt.figure(figsize=(11, 4))

# Raw outer+holes
plt.plot(ensure_closed(outer_loop)[:, 0], ensure_closed(outer_loop)[:, 1], ".", ms=2, label="raw outer")
for i, h in enumerate(hole_loops):
    h2 = ensure_closed(h)
    plt.plot(h2[:, 0], h2[:, 1], ".", ms=2, label="raw hole" if i == 0 else None)

# Deformed outer+holes
plt.plot(ensure_closed(outer_def)[:, 0], ensure_closed(outer_def)[:, 1], "-", lw=2, label="RBF outer")
for i, h in enumerate(holes_def):
    h2 = ensure_closed(h)
    plt.plot(h2[:, 0], h2[:, 1], "-", lw=1.5, label="RBF hole" if i == 0 else None)

plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Boundary smoothing (rho>={rho_thr}): RDP + spline + RBF (outer + holes)")
plt.legend()
plt.tight_layout()
plt.show()



# thickness = 80 #mm

# rho_threshold = 0.90


# import pyvista as pv

# topo_dim = refinedDomain.topology.dim
# cells, cell_types, points = plot.vtk_mesh(refinedDomain, topo_dim)

# grid = pv.UnstructuredGrid(cells, cell_types, points)

# # DG0 has one value per cell
# grid.cell_data["rho"] = rho_refined.x.array.real

# solid = grid.threshold(value=rho_threshold, scalars="rho")

# if solid.n_cells == 0:
#     raise RuntimeError(
#         f"No cells found with rho >= {rho_threshold}. "
#         "Try lowering threshold (e.g. 0.95)."
#     )

# surf = solid.extract_surface().triangulate()

# import vtk

# def vtk_extrude_cap(poly: pv.PolyData, thickness: float) -> pv.PolyData:
#     f = vtk.vtkLinearExtrusionFilter()
#     f.SetInputData(poly)
#     f.SetExtrusionTypeToVectorExtrusion()
#     f.SetVector(0.0, 0.0, float(thickness))
#     f.SetCapping(True)  # <-- makes top/bottom caps
#     f.Update()
#     return pv.wrap(f.GetOutput()).triangulate().clean()

# body = vtk_extrude_cap(surf, thickness)

# body = body.fill_holes(1e6).triangulate().clean()

# body = body.connectivity(largest=True)


# if domain.comm.rank == 0:
#     body.save(path_stl)
#     print(f"Wrote {path_stl} (triangles: {body.n_cells})")
