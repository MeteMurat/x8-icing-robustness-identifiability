import argparse, csv, json, math, os, struct, hashlib
from collections import defaultdict, deque

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def is_binary_stl(path):
    size = os.path.getsize(path)
    if size < 84:
        return False
    with open(path, "rb") as f:
        head = f.read(84)
    n = struct.unpack("<I", head[80:84])[0]
    return 84 + 50*n == size

def read_binary_stl(path):
    triangles = []
    with open(path, "rb") as f:
        header = f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            raw = f.read(50)
            if len(raw) != 50:
                raise ValueError("Unexpected EOF in binary STL")
            vals = struct.unpack("<12fH", raw)
            # vals[0:3] is stored normal; geometry audit recomputes normals from vertices.
            v1 = tuple(float(x) for x in vals[3:6])
            v2 = tuple(float(x) for x in vals[6:9])
            v3 = tuple(float(x) for x in vals[9:12])
            triangles.append((v1,v2,v3))
    return triangles

def read_ascii_stl(path):
    triangles = []
    verts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip().split()
            if len(s) >= 4 and s[0].lower() == "vertex":
                verts.append((float(s[1]), float(s[2]), float(s[3])))
                if len(verts) == 3:
                    triangles.append(tuple(verts))
                    verts = []
    if verts:
        raise ValueError("Incomplete triangle in ASCII STL")
    return triangles

def vsub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def norm(a): return math.sqrt(dot(a,a))

def qkey(v, tol):
    return (int(round(v[0]/tol)), int(round(v[1]/tol)), int(round(v[2]/tol)))

def audit_stl(path):
    fmt = "binary" if is_binary_stl(path) else "ascii"
    tris = read_binary_stl(path) if fmt == "binary" else read_ascii_stl(path)
    if not tris:
        raise ValueError("No triangles")

    pts = [p for tri in tris for p in tri]
    finite = all(math.isfinite(c) for p in pts for c in p)
    if not finite:
        raise ValueError("Non-finite coordinates detected")

    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; zs=[p[2] for p in pts]
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); zmin,zmax=min(zs),max(zs)
    dx,dy,dz=xmax-xmin,ymax-ymin,zmax-zmin
    diag=math.sqrt(dx*dx+dy*dy+dz*dz)
    tol=max(diag*1e-9,1e-12)

    vertex_id={}
    vertices=[]
    def vid(v):
        k=qkey(v,tol)
        if k not in vertex_id:
            vertex_id[k]=len(vertices)
            vertices.append(v)
        return vertex_id[k]

    indexed=[]
    areas=[]
    signed_v6=0.0
    degenerate=0
    tri_keys=defaultdict(int)

    for tri in tris:
        a,b,c=tri
        ia,ib,ic=vid(a),vid(b),vid(c)
        indexed.append((ia,ib,ic))

        ab=vsub(b,a); ac=vsub(c,a)
        cr=cross(ab,ac)
        ar=0.5*norm(cr)
        areas.append(ar)
        if ar <= max(diag*diag*1e-18, 1e-24):
            degenerate += 1

        signed_v6 += dot(a, cross(b,c))
        tri_keys[tuple(sorted((ia,ib,ic)))] += 1

    duplicate_triangles=sum(v-1 for v in tri_keys.values() if v>1)

    edge_uses=defaultdict(list)
    adjacency=[set() for _ in indexed]
    orientation_conflicts=0

    for ti,(a,b,c) in enumerate(indexed):
        for u,v in ((a,b),(b,c),(c,a)):
            key=(u,v) if u<v else (v,u)
            direction=1 if (u,v)==key else -1
            edge_uses[key].append((ti,direction))

    boundary_edges=0
    manifold_edges=0
    nonmanifold_edges=0

    for key,uses in edge_uses.items():
        n=len(uses)
        if n==1:
            boundary_edges += 1
        elif n==2:
            manifold_edges += 1
            (t1,d1),(t2,d2)=uses
            adjacency[t1].add(t2); adjacency[t2].add(t1)
            if d1 == d2:
                orientation_conflicts += 1
        else:
            nonmanifold_edges += 1
            tids=[x[0] for x in uses]
            for i in range(len(tids)):
                for j in range(i+1,len(tids)):
                    adjacency[tids[i]].add(tids[j]); adjacency[tids[j]].add(tids[i])

    # Triangle connected components via shared topological edges.
    seen=[False]*len(indexed)
    comp_sizes=[]
    for s in range(len(indexed)):
        if seen[s]:
            continue
        q=deque([s]); seen[s]=True; n=0
        while q:
            t=q.popleft(); n+=1
            for nb in adjacency[t]:
                if not seen[nb]:
                    seen[nb]=True; q.append(nb)
        comp_sizes.append(n)
    comp_sizes.sort(reverse=True)

    surface_area=sum(areas)
    watertight=(boundary_edges==0 and nonmanifold_edges==0)
    orient_consistent=(orientation_conflicts==0)

    # Volume is only scientifically reported for a closed, manifold,
    # orientation-consistent topology. STL unit remains unspecified.
    volume = None
    if watertight and orient_consistent:
        volume=abs(signed_v6/6.0)

    # Euler characteristic of the welded topology.
    V=len(vertices); E=len(edge_uses); F=len(indexed)
    chi=V-E+F

    cfd_ready_topology=(watertight and orient_consistent and degenerate==0 and duplicate_triangles==0 and len(comp_sizes)==1)

    return {
        "FileName": os.path.basename(path),
        "Format": fmt,
        "FileBytes": os.path.getsize(path),
        "SHA256": sha256_file(path),
        "Triangles": F,
        "WeldedVertices": V,
        "UniqueEdges": E,
        "ConnectedComponents": len(comp_sizes),
        "LargestComponentTriangles": comp_sizes[0] if comp_sizes else 0,
        "BoundaryEdges": boundary_edges,
        "NonManifoldEdges": nonmanifold_edges,
        "OrientationConflictEdges": orientation_conflicts,
        "DegenerateTriangles": degenerate,
        "DuplicateTriangles": duplicate_triangles,
        "EulerCharacteristic": chi,
        "Xmin": xmin, "Xmax": xmax, "SpanX": dx,
        "Ymin": ymin, "Ymax": ymax, "SpanY": dy,
        "Zmin": zmin, "Zmax": zmax, "SpanZ": dz,
        "BoundingDiagonal": diag,
        "WeldToleranceCoordinateUnits": tol,
        "SurfaceArea_CoordUnits2": surface_area,
        "Watertight": watertight,
        "OrientationConsistent": orient_consistent,
        "ClosedVolume_CoordUnits3": volume,
        "CFDReadyTopology": cfd_ready_topology,
        "UnitAssumption": "NONE_STL_HAS_NO_INTRINSIC_UNITS",
        "ComponentTriangleCounts": comp_sizes,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ice-root", required=True)
    ap.add_argument("--output", required=True)
    args=ap.parse_args()

    files=sorted(
        os.path.join(args.ice_root,f)
        for f in os.listdir(args.ice_root)
        if f.lower().endswith(".stl")
    )

    records=[]
    errors=[]
    for p in files:
        try:
            records.append(audit_stl(p))
        except Exception as e:
            errors.append({"FileName": os.path.basename(p), "Error": repr(e)})

    os.makedirs(args.output, exist_ok=True)

    scalar_keys=[
        "FileName","Format","FileBytes","SHA256","Triangles","WeldedVertices","UniqueEdges",
        "ConnectedComponents","LargestComponentTriangles","BoundaryEdges","NonManifoldEdges",
        "OrientationConflictEdges","DegenerateTriangles","DuplicateTriangles","EulerCharacteristic",
        "Xmin","Xmax","SpanX","Ymin","Ymax","SpanY","Zmin","Zmax","SpanZ","BoundingDiagonal",
        "WeldToleranceCoordinateUnits","SurfaceArea_CoordUnits2","Watertight","OrientationConsistent",
        "ClosedVolume_CoordUnits3","CFDReadyTopology","UnitAssumption"
    ]

    with open(os.path.join(args.output,"P0_A5_STL_GEOMETRY_TOPOLOGY_AUDIT.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=scalar_keys)
        w.writeheader()
        for r in records:
            w.writerow({k:r.get(k) for k in scalar_keys})

    with open(os.path.join(args.output,"P0_A5_STL_COMPONENT_SIZES.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f)
        w.writerow(["FileName","ComponentRank","TriangleCount"])
        for r in records:
            for i,n in enumerate(r["ComponentTriangleCounts"],1):
                w.writerow([r["FileName"],i,n])

    with open(os.path.join(args.output,"P0_A5_STL_AUDIT.json"),"w",encoding="utf-8") as f:
        json.dump({"records":records,"errors":errors},f,indent=2)

    with open(os.path.join(args.output,"P0_A5_STL_PARSE_ERRORS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["FileName","Error"])
        w.writeheader()
        w.writerows(errors)

    print("P0_A5_PYTHON_SUMMARY")
    print("files_requested =", len(files))
    print("files_parsed    =", len(records))
    print("parse_errors    =", len(errors))
    print("")
    for r in records:
        print(
            "{0}: tris={1}, verts={2}, comps={3}, boundary={4}, nonmanifold={5}, "
            "orient_conflicts={6}, degenerate={7}, duplicate={8}, watertight={9}, cfd_ready={10}, "
            "area={11:.9g}, spans=({12:.9g},{13:.9g},{14:.9g})".format(
                r["FileName"],r["Triangles"],r["WeldedVertices"],r["ConnectedComponents"],
                r["BoundaryEdges"],r["NonManifoldEdges"],r["OrientationConflictEdges"],
                r["DegenerateTriangles"],r["DuplicateTriangles"],r["Watertight"],
                r["CFDReadyTopology"],r["SurfaceArea_CoordUnits2"],
                r["SpanX"],r["SpanY"],r["SpanZ"]
            )
        )

if __name__ == "__main__":
    main()
