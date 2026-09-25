import argparse, csv, json, math, os, struct
from collections import defaultdict, deque

def is_binary_stl(path):
    size=os.path.getsize(path)
    if size < 84:
        return False
    with open(path,"rb") as f:
        h=f.read(84)
    n=struct.unpack("<I",h[80:84])[0]
    return 84+50*n == size

def read_stl(path):
    tris=[]
    if is_binary_stl(path):
        with open(path,"rb") as f:
            f.read(80)
            n=struct.unpack("<I",f.read(4))[0]
            for _ in range(n):
                vals=struct.unpack("<12fH",f.read(50))
                tris.append((
                    tuple(float(x) for x in vals[3:6]),
                    tuple(float(x) for x in vals[6:9]),
                    tuple(float(x) for x in vals[9:12])
                ))
    else:
        vs=[]
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                s=line.strip().split()
                if len(s)>=4 and s[0].lower()=="vertex":
                    vs.append((float(s[1]),float(s[2]),float(s[3])))
                    if len(vs)==3:
                        tris.append(tuple(vs)); vs=[]
    return tris

def sub(a,b): return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def norm(a): return math.sqrt(dot(a,a))
def qkey(v,tol): return tuple(int(round(c/tol)) for c in v)

def bbox_iou(a,b):
    inter=[]
    for lo1,hi1,lo2,hi2 in zip(a[:3],a[3:],b[:3],b[3:]):
        inter.append(max(0.0,min(hi1,hi2)-max(lo1,lo2)))
    vi=inter[0]*inter[1]*inter[2]
    va=max(0,a[3]-a[0])*max(0,a[4]-a[1])*max(0,a[5]-a[2])
    vb=max(0,b[3]-b[0])*max(0,b[4]-b[1])*max(0,b[5]-b[2])
    u=va+vb-vi
    return vi/u if u>0 else 0.0

def component_metrics(tris, ids, vertices, comp_tri_indices):
    pts=[]
    area=0.0
    signed6=0.0
    edges=defaultdict(list)
    deg=0

    for ti in comp_tri_indices:
        ia,ib,ic=ids[ti]
        a,b,c=vertices[ia],vertices[ib],vertices[ic]
        pts.extend((a,b,c))
        cr=cross(sub(b,a),sub(c,a))
        ar=0.5*norm(cr)
        area += ar
        if ar <= 1e-24: deg += 1
        signed6 += dot(a,cross(b,c))
        for u,v in ((ia,ib),(ib,ic),(ic,ia)):
            k=(u,v) if u<v else (v,u)
            d=1 if (u,v)==k else -1
            edges[k].append(d)

    unique_vids=set()
    for ti in comp_tri_indices:
        unique_vids.update(ids[ti])

    boundary=sum(1 for x in edges.values() if len(x)==1)
    nonman=sum(1 for x in edges.values() if len(x)>2)
    orient=sum(1 for x in edges.values() if len(x)==2 and x[0]==x[1])

    p=[vertices[i] for i in unique_vids]
    xs=[x[0] for x in p]; ys=[x[1] for x in p]; zs=[x[2] for x in p]
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys); zmin,zmax=min(zs),max(zs)
    cx=sum(xs)/len(xs); cy=sum(ys)/len(ys); cz=sum(zs)/len(zs)
    closed=(boundary==0 and nonman==0)
    consistent=(orient==0)
    volume=abs(signed6/6.0) if closed and consistent else None

    return {
        "Triangles":len(comp_tri_indices),
        "Vertices":len(unique_vids),
        "BoundaryEdges":boundary,
        "NonManifoldEdges":nonman,
        "OrientationConflictEdges":orient,
        "DegenerateTriangles":deg,
        "Watertight":closed,
        "OrientationConsistent":consistent,
        "SurfaceArea_CoordUnits2":area,
        "ClosedVolume_CoordUnits3":volume,
        "Xmin":xmin,"Xmax":xmax,"SpanX":xmax-xmin,
        "Ymin":ymin,"Ymax":ymax,"SpanY":ymax-ymin,
        "Zmin":zmin,"Zmax":zmax,"SpanZ":zmax-zmin,
        "VertexCentroidX":cx,"VertexCentroidY":cy,"VertexCentroidZ":cz,
        "_bbox":(xmin,ymin,zmin,xmax,ymax,zmax)
    }

def audit(path):
    tris=read_stl(path)
    pts=[p for t in tris for p in t]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; zs=[p[2] for p in pts]
    dx=max(xs)-min(xs); dy=max(ys)-min(ys); dz=max(zs)-min(zs)
    diag=math.sqrt(dx*dx+dy*dy+dz*dz)
    tol=max(diag*1e-9,1e-12)

    vmap={}; vertices=[]
    def vid(v):
        k=qkey(v,tol)
        if k not in vmap:
            vmap[k]=len(vertices); vertices.append(v)
        return vmap[k]

    ids=[]
    edge_to_tris=defaultdict(list)
    for ti,t in enumerate(tris):
        a,b,c=(vid(t[0]),vid(t[1]),vid(t[2]))
        ids.append((a,b,c))
        for u,v in ((a,b),(b,c),(c,a)):
            k=(u,v) if u<v else (v,u)
            edge_to_tris[k].append(ti)

    adj=[set() for _ in tris]
    for uses in edge_to_tris.values():
        if len(uses)>=2:
            for i in range(len(uses)):
                for j in range(i+1,len(uses)):
                    adj[uses[i]].add(uses[j]); adj[uses[j]].add(uses[i])

    comps=[]
    seen=[False]*len(tris)
    for s in range(len(tris)):
        if seen[s]: continue
        q=deque([s]); seen[s]=True; c=[]
        while q:
            t=q.popleft(); c.append(t)
            for nb in adj[t]:
                if not seen[nb]:
                    seen[nb]=True; q.append(nb)
        comps.append(c)

    metrics=[component_metrics(tris,ids,vertices,c) for c in comps]
    metrics.sort(key=lambda x:(x["ClosedVolume_CoordUnits3"] or 0.0), reverse=True)

    for rank,m in enumerate(metrics,1):
        m["FileName"]=os.path.basename(path)
        m["ComponentRankByVolume"]=rank
        m["ComponentID"]="C{}".format(rank)
        m["VolumeFractionOfFile"]=None
        m["AreaFractionOfFile"]=None
        m["SemanticLabel"]="UNRESOLVED_COMPONENT"

    totalv=sum((m["ClosedVolume_CoordUnits3"] or 0.0) for m in metrics)
    totala=sum(m["SurfaceArea_CoordUnits2"] for m in metrics)
    for m in metrics:
        if totalv>0: m["VolumeFractionOfFile"]=(m["ClosedVolume_CoordUnits3"] or 0.0)/totalv
        if totala>0: m["AreaFractionOfFile"]=m["SurfaceArea_CoordUnits2"]/totala

    pair={}
    if len(metrics)==2:
        a,b=metrics
        dc=math.sqrt(
            (a["VertexCentroidX"]-b["VertexCentroidX"])**2+
            (a["VertexCentroidY"]-b["VertexCentroidY"])**2+
            (a["VertexCentroidZ"]-b["VertexCentroidZ"])**2
        )
        pair={
            "FileName":os.path.basename(path),
            "Component1":"C1",
            "Component2":"C2",
            "CentroidDistance_CoordUnits":dc,
            "CentroidDistance_over_FileDiag":dc/diag if diag>0 else None,
            "BBoxIoU":bbox_iou(a["_bbox"],b["_bbox"]),
            "VolumeRatio_C1_over_C2":(
                a["ClosedVolume_CoordUnits3"]/b["ClosedVolume_CoordUnits3"]
                if b["ClosedVolume_CoordUnits3"] not in (None,0) else None
            ),
            "AreaRatio_C1_over_C2":(
                a["SurfaceArea_CoordUnits2"]/b["SurfaceArea_CoordUnits2"]
                if b["SurfaceArea_CoordUnits2"]!=0 else None
            ),
            "Interpretation":"UNRESOLVED_TWO_CLOSED_COMPONENTS"
        }

    for m in metrics: m.pop("_bbox",None)
    return metrics,pair

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ice-root",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()

    files=sorted(os.path.join(a.ice_root,f) for f in os.listdir(a.ice_root) if f.lower().endswith(".stl"))
    rows=[]; pairs=[]; errors=[]

    for p in files:
        try:
            m,pair=audit(p)
            rows.extend(m)
            if pair: pairs.append(pair)
        except Exception as e:
            errors.append({"FileName":os.path.basename(p),"Error":repr(e)})

    fields=[
        "FileName","ComponentID","ComponentRankByVolume","SemanticLabel",
        "Triangles","Vertices","BoundaryEdges","NonManifoldEdges","OrientationConflictEdges",
        "DegenerateTriangles","Watertight","OrientationConsistent",
        "SurfaceArea_CoordUnits2","ClosedVolume_CoordUnits3","AreaFractionOfFile","VolumeFractionOfFile",
        "Xmin","Xmax","SpanX","Ymin","Ymax","SpanY","Zmin","Zmax","SpanZ",
        "VertexCentroidX","VertexCentroidY","VertexCentroidZ"
    ]

    with open(os.path.join(a.output,"P0_A5R1_COMPONENT_METRICS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    pfields=[
        "FileName","Component1","Component2","CentroidDistance_CoordUnits",
        "CentroidDistance_over_FileDiag","BBoxIoU","VolumeRatio_C1_over_C2",
        "AreaRatio_C1_over_C2","Interpretation"
    ]
    with open(os.path.join(a.output,"P0_A5R1_COMPONENT_PAIR_RELATIONS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=pfields); w.writeheader(); w.writerows(pairs)

    with open(os.path.join(a.output,"P0_A5R1_COMPONENT_AUDIT.json"),"w",encoding="utf-8") as f:
        json.dump({"components":rows,"pairs":pairs,"errors":errors},f,indent=2)

    with open(os.path.join(a.output,"P0_A5R1_PARSE_ERRORS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["FileName","Error"]); w.writeheader(); w.writerows(errors)

    print("P0_A5R1_COMPONENT_SUMMARY")
    print("stl_files       =",len(files))
    print("components      =",len(rows))
    print("parse_errors    =",len(errors))
    print("")
    byfile={}
    for r in rows:
        byfile.setdefault(r["FileName"],[]).append(r)
    for fn in sorted(byfile):
        rs=sorted(byfile[fn],key=lambda x:x["ComponentRankByVolume"])
        print(fn)
        for r in rs:
            print(
                "  {0}: tris={1}, verts={2}, area={3:.9g}, volume={4:.9g}, "
                "vol_frac={5:.6f}, spans=({6:.9g},{7:.9g},{8:.9g}), "
                "watertight={9}, orient_ok={10}".format(
                    r["ComponentID"],r["Triangles"],r["Vertices"],
                    r["SurfaceArea_CoordUnits2"],r["ClosedVolume_CoordUnits3"],
                    r["VolumeFractionOfFile"] or 0.0,
                    r["SpanX"],r["SpanY"],r["SpanZ"],
                    r["Watertight"],r["OrientationConsistent"]
                )
            )
        pp=[x for x in pairs if x["FileName"]==fn]
        if pp:
            q=pp[0]
            print(
                "  pair: centroid_dist/file_diag={0:.6f}, bbox_iou={1:.6f}, "
                "vol_ratio_C1/C2={2:.6f}, area_ratio_C1/C2={3:.6f}".format(
                    q["CentroidDistance_over_FileDiag"],q["BBoxIoU"],
                    q["VolumeRatio_C1_over_C2"],q["AreaRatio_C1_over_C2"]
                )
            )

if __name__=="__main__":
    main()
