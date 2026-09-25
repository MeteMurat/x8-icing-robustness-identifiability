import argparse,csv,json,math,os,re,statistics
from collections import defaultdict

EPS=1e-12

def F(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None

def qtile(xs,p):
    xs=sorted(x for x in xs if x is not None and math.isfinite(x))
    if not xs:return None
    if len(xs)==1:return xs[0]
    pos=(len(xs)-1)*p
    lo=int(math.floor(pos));hi=int(math.ceil(pos))
    if lo==hi:return xs[lo]
    w=pos-lo
    return xs[lo]*(1-w)+xs[hi]*w

def mean(xs): return sum(xs)/len(xs) if xs else None

def sd(xs):
    if len(xs)<2:return 0.0
    m=mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

def support_indices(path):
    s=defaultdict(set)
    with open(path,"r",newline="",encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            s[(r["configuration"],r["file_name"])].add(int(r["sample_index"]))
    return s

def training_files(folder):
    out=[]
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".csv"):continue
        m=re.search(r"_ID_(\d+)\.csv$",fn,re.I)
        if m and int(m.group(1))<=3:
            out.append((fn,os.path.join(folder,fn)))
    return out

def axis_of(fn): return "lon" if fn.lower().startswith("lon_") else "lat"

def row_features(row,axis,b,c):
    Va=F(row.get("Va_EKF_CG"))
    alpha=F(row.get("alpha"))
    beta=F(row.get("beta"))
    p=F(row.get("p"));q=F(row.get("q"));r=F(row.get("r"))
    elevator=F(row.get("elevator"));aileron=F(row.get("aileron"))
    if Va is None or Va<=1e-8:return None

    if axis=="lon":
        vals={
            "Va":Va,
            "alpha":alpha,
            "q_hat":None if q is None else q*c/(2*Va),
            "delta_e":elevator
        }
    else:
        vals={
            "Va":Va,
            "alpha":alpha,
            "beta":beta,
            "p_hat":None if p is None else p*b/(2*Va),
            "r_hat":None if r is None else r*b/(2*Va),
            "delta_a":aileron
        }
    if any(v is None or not math.isfinite(v) for v in vals.values()):
        return None
    return vals

def matrix_rank(A,tol=1e-10):
    if not A:return 0
    M=[list(map(float,row)) for row in A]
    n=len(M);m=len(M[0]);rank=0;col=0
    while rank<n and col<m:
        piv=max(range(rank,n),key=lambda r:abs(M[r][col]))
        if abs(M[piv][col])<=tol:
            col+=1;continue
        M[rank],M[piv]=M[piv],M[rank]
        d=M[rank][col]
        for j in range(col,m):M[rank][j]/=d
        for r in range(n):
            if r==rank:continue
            q=M[r][col]
            if abs(q)<=tol:continue
            for j in range(col,m):M[r][j]-=q*M[rank][j]
        rank+=1;col+=1
    return rank

def invert(A):
    n=len(A)
    M=[list(map(float,A[i]))+[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
    for k in range(n):
        piv=max(range(k,n),key=lambda r:abs(M[r][k]))
        if abs(M[piv][k])<1e-12:raise ArithmeticError("singular")
        M[k],M[piv]=M[piv],M[k]
        d=M[k][k]
        for j in range(2*n):M[k][j]/=d
        for r in range(n):
            if r==k:continue
            q=M[r][k]
            for j in range(2*n):M[r][j]-=q*M[k][j]
    return [row[n:] for row in M]

def jacobi_eigenvalues(A,max_iter=200,tol=1e-12):
    n=len(A)
    M=[list(map(float,row)) for row in A]
    if n==1:return [M[0][0]]
    for _ in range(max_iter):
        p=q=0;mx=0.0
        for i in range(n):
            for j in range(i+1,n):
                if abs(M[i][j])>mx:
                    mx=abs(M[i][j]);p=i;q=j
        if mx<tol:break
        app=M[p][p];aqq=M[q][q];apq=M[p][q]
        phi=0.5*math.atan2(2*apq,aqq-app)
        c=math.cos(phi);s=math.sin(phi)
        for k in range(n):
            if k in (p,q):continue
            mkp=M[k][p];mkq=M[k][q]
            M[k][p]=M[p][k]=c*mkp-s*mkq
            M[k][q]=M[q][k]=s*mkp+c*mkq
        M[p][p]=c*c*app-2*s*c*apq+s*s*aqq
        M[q][q]=s*s*app+2*s*c*apq+c*c*aqq
        M[p][q]=M[q][p]=0.0
    return sorted(M[i][i] for i in range(n))

def design_metrics(rows,predictors):
    n=len(rows);p=len(predictors)
    if n<2:
        return {"n":n,"predictors":p,"rank":0,"condition_number":None,"max_vif":None,"max_abs_pair_corr":None}

    cols={k:[r[k] for r in rows] for k in predictors}
    mus={k:mean(cols[k]) for k in predictors}
    sds={k:sd(cols[k]) for k in predictors}
    if any(sds[k]<=1e-12 for k in predictors):
        return {"n":n,"predictors":p,"rank":0,"condition_number":float("inf"),"max_vif":float("inf"),"max_abs_pair_corr":1.0}

    Z=[[ (r[k]-mus[k])/sds[k] for k in predictors] for r in rows]

    R=[]
    for i in range(p):
        row=[]
        for j in range(p):
            row.append(sum(z[i]*z[j] for z in Z)/(n-1))
        R.append(row)

    eig=jacobi_eigenvalues(R)
    pos=[x for x in eig if x>1e-10]
    rank=len(pos)
    cond=math.sqrt(max(pos)/min(pos)) if pos and rank==p else float("inf")

    try:
        inv=invert(R)
        vifs=[inv[i][i] for i in range(p)]
        maxv=max(vifs)
    except Exception:
        maxv=float("inf")

    maxcorr=0.0
    for i in range(p):
        for j in range(i+1,p):
            maxcorr=max(maxcorr,abs(R[i][j]))

    return {
        "n":n,
        "predictors":p,
        "rank":rank,
        "condition_number":cond,
        "max_vif":maxv,
        "max_abs_pair_corr":maxcorr,
        "min_corr_eigenvalue":min(eig),
        "max_corr_eigenvalue":max(eig)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-dir",required=True)
    ap.add_argument("--iced-dir",required=True)
    ap.add_argument("--pilot-samples",required=True)
    ap.add_argument("--clean-json",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    with open(a.clean_json,"r",encoding="utf-8-sig") as fh:
        p=json.load(fh)
    b=float(p["b"]);c=float(p["c"])

    support=support_indices(a.pilot_samples)

    datasets=[]
    pair_names={"lon":set(),"lat":set()}
    errors=[]

    for cfg,folder in [("clean",a.clean_dir),("iced",a.iced_dir)]:
        for fn,path in training_files(folder):
            axis=axis_of(fn)
            pair_names[axis].add(fn)
            try:
                with open(path,"r",newline="",encoding="utf-8-sig") as fh:
                    rows=list(csv.DictReader(fh))
                s=support.get((cfg,fn),set())
                for idx,row in enumerate(rows):
                    if idx not in s:continue
                    feat=row_features(row,axis,b,c)
                    if feat is None:continue
                    datasets.append({"configuration":cfg,"file_name":fn,"axis":axis,**feat})
            except Exception as e:
                errors.append({"configuration":cfg,"file_name":fn,"error":repr(e)})

    envelope_vars={
        "lon":["Va","alpha","q_hat","delta_e"],
        "lat":["Va","alpha","beta","p_hat","r_hat","delta_a"]
    }
    design_vars={
        "lon":["alpha","q_hat","delta_e"],
        "lat":["beta","p_hat","r_hat","delta_a"]
    }

    envelopes={}
    envelope_rows=[]

    for axis in ("lon","lat"):
        envelopes[axis]={}
        for var in envelope_vars[axis]:
            cvals=[r[var] for r in datasets if r["axis"]==axis and r["configuration"]=="clean"]
            ivals=[r[var] for r in datasets if r["axis"]==axis and r["configuration"]=="iced"]
            c_lo=qtile(cvals,.025);c_hi=qtile(cvals,.975)
            i_lo=qtile(ivals,.025);i_hi=qtile(ivals,.975)
            lo=max(c_lo,i_lo);hi=min(c_hi,i_hi)
            width=hi-lo
            envelopes[axis][var]=(lo,hi)
            envelope_rows.append({
                "axis":axis,"variable":var,
                "clean_q025":c_lo,"clean_q975":c_hi,
                "iced_q025":i_lo,"iced_q975":i_hi,
                "common_low":lo,"common_high":hi,"common_width":width,
                "positive_width":width>0
            })

    retained=[]
    for r in datasets:
        ok=True
        for var,(lo,hi) in envelopes[r["axis"]].items():
            if not (lo<=r[var]<=hi):
                ok=False;break
        if ok:retained.append(r)

    metrics=[]
    gate_parts=[]
    for axis in ("lon","lat"):
        for cfg in ("clean","iced"):
            before=[r for r in datasets if r["axis"]==axis and r["configuration"]==cfg]
            after=[r for r in retained if r["axis"]==axis and r["configuration"]==cfg]
            dm=design_metrics(after,design_vars[axis])
            retention=len(after)/len(before) if before else 0.0
            row={
                "axis":axis,"configuration":cfg,
                "rows_before_envelope":len(before),
                "rows_in_common_envelope":len(after),
                "retention_fraction":retention,
                **dm
            }
            metrics.append(row)

            part=(
                len(after)>=1000 and
                retention>=0.50 and
                dm["rank"]==len(design_vars[axis]) and
                dm["condition_number"]<=15.0 and
                dm["max_vif"]<=10.0 and
                dm["max_abs_pair_corr"]<=0.95
            )
            gate_parts.append((axis,cfg,part))

    # Pair representation on common envelope.
    pair_rows=[]
    for axis in ("lon","lat"):
        names=sorted(pair_names[axis])
        for fn in names:
            c_before=sum(1 for r in datasets if r["configuration"]=="clean" and r["file_name"]==fn)
            i_before=sum(1 for r in datasets if r["configuration"]=="iced" and r["file_name"]==fn)
            c_after=sum(1 for r in retained if r["configuration"]=="clean" and r["file_name"]==fn)
            i_after=sum(1 for r in retained if r["configuration"]=="iced" and r["file_name"]==fn)
            represented=(c_after>=25 and i_after>=25)
            pair_rows.append({
                "axis":axis,"file_name":fn,
                "clean_before":c_before,"iced_before":i_before,
                "clean_common_envelope":c_after,"iced_common_envelope":i_after,
                "pair_represented":represented
            })

    axis_rep={}
    for axis in ("lon","lat"):
        rr=[r for r in pair_rows if r["axis"]==axis]
        n=sum(1 for r in rr if r["pair_represented"])
        axis_rep[axis]={
            "pairs":len(rr),
            "represented_pairs":n,
            "represented_fraction":n/len(rr) if rr else 0.0
        }

    envelope_width_pass=all(r["positive_width"] for r in envelope_rows)
    design_pass=all(x[2] for x in gate_parts)
    representation_pass=all(axis_rep[a]["represented_fraction"]>=0.80 for a in ("lon","lat"))
    global_pass=(
        len(errors)==0 and
        envelope_width_pass and
        design_pass and
        representation_pass
    )

    with open(os.path.join(a.out,"P1_A3B0_COMMON_ENVELOPE.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        fields=["axis","variable","clean_q025","clean_q975","iced_q025","iced_q975",
                "common_low","common_high","common_width","positive_width"]
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(envelope_rows)

    with open(os.path.join(a.out,"P1_A3B0_DESIGN_IDENTIFIABILITY.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        fields=["axis","configuration","rows_before_envelope","rows_in_common_envelope","retention_fraction",
                "n","predictors","rank","condition_number","max_vif","max_abs_pair_corr",
                "min_corr_eigenvalue","max_corr_eigenvalue"]
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(metrics)

    with open(os.path.join(a.out,"P1_A3B0_PAIR_REPRESENTATION.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        fields=["axis","file_name","clean_before","iced_before","clean_common_envelope","iced_common_envelope","pair_represented"]
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(pair_rows)

    with open(os.path.join(a.out,"P1_A3B0_PARSE_ERRORS.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=["configuration","file_name","error"]);w.writeheader();w.writerows(errors)

    result={
        "scope":{
            "identification_only":True,
            "validation_files_used":0,
            "aerodynamic_outcome_columns_used":0,
            "supported_rows_total":len(datasets),
            "common_envelope_rows_total":len(retained)
        },
        "models":{
            "longitudinal":{
                "common_envelope_variables":envelope_vars["lon"],
                "design_predictors":design_vars["lon"],
                "future_primary_outcomes":["C_L_lift","C_m_pitch"],
                "future_drag_analysis":"separate predeclared model; not fit at this stage"
            },
            "lateral":{
                "common_envelope_variables":envelope_vars["lat"],
                "design_predictors":design_vars["lat"],
                "future_primary_outcomes":["C_Y_wind","C_l_roll","C_n_yaw"]
            }
        },
        "frozen_envelope":{
            "quantiles":"configuration-specific 2.5th-97.5th percentiles, intersected variable-by-variable",
            "row_rule":"retain a row only if every axis-specific common-envelope variable lies inside the intersection"
        },
        "predeclared_identifiability_gate":{
            "minimum_rows_per_axis_configuration":1000,
            "minimum_retention_fraction":0.50,
            "full_rank_required":True,
            "maximum_condition_number":15.0,
            "maximum_vif":10.0,
            "maximum_abs_pairwise_correlation":0.95,
            "minimum_pair_rows_each_configuration":25,
            "minimum_axis_pair_representation_fraction":0.80
        },
        "design_metrics":metrics,
        "pair_representation":axis_rep,
        "envelope_width_pass":envelope_width_pass,
        "design_identifiability_pass":design_pass,
        "pair_representation_pass":representation_pass,
        "parse_errors":len(errors),
        "global_pass":global_pass
    }

    with open(os.path.join(a.out,"P1_A3B0_IDENTIFIABILITY_AUDIT.json"),"w",encoding="utf-8") as fh:
        json.dump(result,fh,indent=2)

    print("P1_A3B0_IDENTIFIABILITY_SUMMARY")
    print("validation_files_used         = 0")
    print("aerodynamic_outcomes_used     = 0")
    print("supported_rows_total          =",len(datasets))
    print("common_envelope_rows_total    =",len(retained))
    print("parse_errors                  =",len(errors))
    print("envelope_width_pass           =",envelope_width_pass)
    print("design_identifiability_pass   =",design_pass)
    print("pair_representation_pass      =",representation_pass)
    print("global_identifiability_gate   =",global_pass)
    print("")
    print("DESIGN METRICS")
    for r in metrics:
        print("  {0:3s} {1:5s}: n={2}, retain={3:.3f}, rank={4}/{5}, cond={6:.3f}, maxVIF={7:.3f}, max|r|={8:.3f}".format(
            r["axis"],r["configuration"],r["rows_in_common_envelope"],r["retention_fraction"],
            r["rank"],r["predictors"],r["condition_number"],r["max_vif"],r["max_abs_pair_corr"]
        ))
    print("")
    print("PAIR REPRESENTATION")
    for axis in ("lon","lat"):
        z=axis_rep[axis]
        print("  {0}: {1}/{2} = {3:.3f}".format(axis,z["represented_pairs"],z["pairs"],z["represented_fraction"]))

if __name__=="__main__":
    main()
