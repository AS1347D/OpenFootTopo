import numpy as np
import pyvista as pv
import gmsh
import sys

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

def footMesh(meshPath, outerLoop, holeLoops, lc = None):

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add(__name__)

    # Outer curve
    outer_points = []
    for x,y in outerLoop:
        outer_points.append(gmsh.model.occ.addPoint(x, y, 0))

    gmsh.model.occ.synchronize()

    outer_lines = []
    for i in range(len(outer_points)):
        if i == len(outer_points)-2:
            outer_lines.append(gmsh.model.occ.addLine(outer_points[i], outer_points[0]))
            break
        else:
            outer_lines.append(gmsh.model.occ.addLine(outer_points[i], outer_points[i+1]))

    outer_curve = gmsh.model.occ.addCurveLoop(outer_lines)

    gmsh.model.occ.synchronize()

    base = gmsh.model.occ.addPlaneSurface([outer_curve])
    gmsh.model.occ.synchronize()


    hole_tags = []
    for hole in holeLoops:
        outer_points = []
        for x,y in hole:
            outer_points.append(gmsh.model.occ.addPoint(x, y, 0))

        gmsh.model.occ.synchronize()

        outer_lines = []
        for i in range(len(outer_points)):
            if i == len(outer_points)-2:
                outer_lines.append(gmsh.model.occ.addLine(outer_points[i], outer_points[0]))
                break
            else:
                outer_lines.append(gmsh.model.occ.addLine(outer_points[i], outer_points[i+1]))

        outer_curve = gmsh.model.occ.addCurveLoop(outer_lines)

        gmsh.model.occ.synchronize()

        hole_tags.append((2, gmsh.model.occ.addPlaneSurface([outer_curve])))
        gmsh.model.occ.synchronize()

    footSurface = gmsh.model.occ.cut([(2, base)], hole_tags) 
    gmsh.model.occ.synchronize()

    # # Physical tags
    line_tags = gmsh.model.getEntities(dim=1)
    for line in line_tags:
        gmsh.model.addPhysicalGroup(1, [line[1]], tag=line[1])

    # # Physical tags
    surface_tags = gmsh.model.getEntities(dim=2)
    for surface in surface_tags:
        gmsh.model.addPhysicalGroup(2, [surface[1]], tag=surface[1])

    if lc:
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc)

    gmsh.option.setNumber("Mesh.Optimize", 1)

    gmsh.model.mesh.generate(2)
    # gmsh.model.mesh.refine()

    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.write("{}.msh".format(meshPath))

    gmsh.finalize()