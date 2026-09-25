import argparse, csv, json, math, os, struct
from collections import defaultdict, deque

def is_binary_stl(path):
    size=os.path.getsize(path)
    if size<84: return False
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
                tris.append((tuple(vals[3:6]),tuple(vals[6:9]),tuple(vals[9:12])))
    else:
        v=[]
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            for line in f:
                s=line.strip().split()
                if len(s)>=4 and s[0].lower()=="vertex":
                    v.append((float(s[1]),float(s[2]),float(s[3])))
                    if len(v)==3:
                        tris.append(tuple(v)); v=[]
    return tris

def sub(a,b): return (a[0]-b[0],a[1]-b[1],a[2]-b[2])
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def norm(a): return math.sqrt(dot(a,a))
def dist(a,b):
    d=sub(a,b)
    return norm(d)

def qkey(v,tol): return tuple(int(round(c/tol)) for c in v)

def split_components(tris):
    pts=[p for t in tris for p in t]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; zs=[p[2] for p in pts]
    diag=math.sqrt((max(xs)-min(xs))**2+(max(ys)-min(ys))**2+(max(zs)-min(zs))**2)
    tol=max(diag*1e-9,1e-12)
    vmap={}; verts=[]
    def vid(v):
        k=qkey(v,tol)
        if k not in vmap:
            vmap[k]=len(verts); verts.append(tuple(float(x) for x in v))
        return vmap[k]
    ids=[]
    e2t=defaultdict(list)
    for ti,t in enumerate(tris):
        a,b,c=vid(t[0]),vid(t[1]),vid(t[2])
        ids.append((a,b,c))
        for u,v in ((a,b),(b,c),(c,a)):
            k=(u,v) if u<v else (v,u)
            e2t[k].append(ti)
    adj=[set() for _ in tris]
    for uses in e2t.values():
        if len(uses)>=2:
            for i in range(len(uses)):
                for j in range(i+1,len(uses)):
                    adj[uses[i]].add(uses[j]); adj[uses[j]].add(uses[i])
    seen=[False]*len(tris); comps=[]
    for s in range(len(tris)):
        if seen[s]: continue
        q=deque([s]); seen[s]=True; cc=[]
        while q:
            t=q.popleft(); cc.append(t)
            for nb in adj[t]:
                if not seen[nb]:
                    seen[nb]=True; q.append(nb)
        comps.append(cc)
    return verts,ids,comps

def comp_metrics(verts,ids,cc):
    vids=set()
    area=0.0
    edges=set()
    tri_areas=[]
    edge_lengths=[]
    for ti in cc:
        a,b,c=ids[ti]
        vids.update((a,b,c))
        A,B,C=verts[a],verts[b],verts[c]
        cr=cross(sub(B,A),sub(C,A))
        ar=0.5*norm(cr)
        area += ar
        tri_areas.append(ar)
        for u,v in ((a,b),(b,c),(c,a)):
            k=(u,v) if u<v else (v,u)
            if k not in edges:
                edges.add(k)
                edge_lengths.append(dist(verts[u],verts[v]))
    P=[verts[i] for i in vids]
    xs=[p[0] for p in P]; ys=[p[1] for p in P]; zs=[p[2] for p in P]
    cx=sum(xs)/len(xs); cy=sum(ys)/len(ys); cz=sum(zs)/len(zs)
    spanx=max(xs)-min(xs); spany=max(ys)-min(ys); spanz=max(zs)-min(zs)
    bboxvol=spanx*spany*spanz
    # coarse tessellation diagnostics
    mean_tri_area=area/len(cc)
    mean_edge=sum(edge_lengths)/len(edge_lengths) if edge_lengths else 0.0
    tri_density=len(cc)/area if area>0 else None
    vertex_density=len(vids)/area if area>0 else None
    return {
        "Triangles":len(cc),
        "Vertices":len(vids),
        "Area_mm2":area,
        "MeanTriangleArea_mm2":mean_tri_area,
        "UniqueEdges":len(edges),
        "MeanEdgeLength_mm":mean_edge,
        "TriangleDensity_per_mm2":tri_density,
        "VertexDensity_per_mm2":vertex_density,
        "SpanX_mm":spanx,"SpanY_mm":spany,"SpanZ_mm":spanz,
        "BBoxVolume_mm3":bboxvol,
        "CentroidX_mm":cx,"CentroidY_mm":cy,"CentroidZ_mm":cz,
    }

def point_bbox_signed_margin(p,b):
    xmin,ymin,zmin,xmax,ymax,zmax=b
    margins=(p[0]-xmin,xmax-p[0],p[1]-ymin,ymax-p[1],p[2]-zmin,zmax-p[2])
    if all(m>=0 for m in margins):
        return min(margins)
    outs=[
        max(xmin-p[0],0,p[0]-xmax),
        max(ymin-p[1],0,p[1]-ymax),
        max(zmin-p[2],0,p[2]-zmax)
    ]
    return -math.sqrt(sum(x*x for x in outs))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ice-root",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()

    rows=[]; relations=[]; errors=[]
    files=sorted(os.path.join(a.ice_root,f) for f in os.listdir(a.ice_root) if f.lower().endswith(".stl"))

    for path in files:
        try:
            tris=read_stl(path)
            verts,ids,comps=split_components(tris)
            cms=[]
            for cc in comps:
                m=comp_metrics(verts,ids,cc)
                cms.append((m,cc))
            # rank by triangle count, because detailed component is the central semantic clue;
            # keep this as an evidence label, not a physical identity label.
            cms.sort(key=lambda x:x[0]["Triangles"],reverse=True)

            for rank,(m,cc) in enumerate(cms,1):
                r={"FileName":os.path.basename(path),"EvidenceComponentID":"E{}".format(rank)}
                r.update(m)
                r["SemanticLabel"]="UNRESOLVED"
                rows.append(r)

            if len(cms)==2:
                m1,cc1=cms[0]; m2,cc2=cms[1]
                ratio_tri=m1["Triangles"]/m2["Triangles"] if m2["Triangles"] else None
                ratio_density=m1["TriangleDensity_per_mm2"]/m2["TriangleDensity_per_mm2"] if m2["TriangleDensity_per_mm2"] else None
                ratio_area=m1["Area_mm2"]/m2["Area_mm2"] if m2["Area_mm2"] else None
                relations.append({
                    "FileName":os.path.basename(path),
                    "DetailedComponentByTriangleCount":"E1",
                    "CoarseComponentByTriangleCount":"E2",
                    "TriangleCountRatio_E1_over_E2":ratio_tri,
                    "TriangleDensityRatio_E1_over_E2":ratio_density,
                    "AreaRatio_E1_over_E2":ratio_area,
                    "CentroidDistance_mm":math.sqrt(
                        (m1["CentroidX_mm"]-m2["CentroidX_mm"])**2+
                        (m1["CentroidY_mm"]-m2["CentroidY_mm"])**2+
                        (m1["CentroidZ_mm"]-m2["CentroidZ_mm"])**2
                    ),
                    "EvidenceInterpretation":"E1 is much more finely tessellated than E2; this supports different geometric provenance/roles but does not by itself identify physical semantics."
                })
        except Exception as e:
            errors.append({"FileName":os.path.basename(path),"Error":repr(e)})

    fields=[
        "FileName","EvidenceComponentID","Triangles","Vertices","Area_mm2","MeanTriangleArea_mm2",
        "UniqueEdges","MeanEdgeLength_mm","TriangleDensity_per_mm2","VertexDensity_per_mm2",
        "SpanX_mm","SpanY_mm","SpanZ_mm","BBoxVolume_mm3",
        "CentroidX_mm","CentroidY_mm","CentroidZ_mm","SemanticLabel"
    ]
    with open(os.path.join(a.output,"P0_A5R4_COMPONENT_COMPLEXITY_METRICS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

    rfields=[
        "FileName","DetailedComponentByTriangleCount","CoarseComponentByTriangleCount",
        "TriangleCountRatio_E1_over_E2","TriangleDensityRatio_E1_over_E2","AreaRatio_E1_over_E2",
        "CentroidDistance_mm","EvidenceInterpretation"
    ]
    with open(os.path.join(a.output,"P0_A5R4_COMPONENT_RELATIONS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=rfields); w.writeheader(); w.writerows(relations)

    with open(os.path.join(a.output,"P0_A5R4_PARSE_ERRORS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["FileName","Error"]); w.writeheader(); w.writerows(errors)

    with open(os.path.join(a.output,"P0_A5R4_GEOMETRIC_EVIDENCE.json"),"w",encoding="utf-8") as f:
        json.dump({"components":rows,"relations":relations,"errors":errors},f,indent=2)

    print("P0_A5R4_GEOMETRIC_EVIDENCE_SUMMARY")
    print("files       =",len(files))
    print("components  =",len(rows))
    print("errors      =",len(errors))
    print("")
    by={}
    for r in rows: by.setdefault(r["FileName"],[]).append(r)
    for fn in sorted(by):
        rs=sorted(by[fn],key=lambda x:x["EvidenceComponentID"])
        print(fn)
        for r in rs:
            print("  {0}: tris={1}, area={2:.6f} mm^2, mean_tri_area={3:.6f} mm^2, tri_density={4:.9f}/mm^2, spans=({5:.3f},{6:.3f},{7:.3f}) mm".format(
                r["EvidenceComponentID"],r["Triangles"],r["Area_mm2"],r["MeanTriangleArea_mm2"],
                r["TriangleDensity_per_mm2"],r["SpanX_mm"],r["SpanY_mm"],r["SpanZ_mm"]
            ))
        rel=[x for x in relations if x["FileName"]==fn][0]
        print("  ratios: triangles={0:.3f}, density={1:.3f}, area={2:.3f}, centroid_distance={3:.3f} mm".format(
            rel["TriangleCountRatio_E1_over_E2"],rel["TriangleDensityRatio_E1_over_E2"],
            rel["AreaRatio_E1_over_E2"],rel["CentroidDistance_mm"]
        ))

if __name__=="__main__":
    main()
