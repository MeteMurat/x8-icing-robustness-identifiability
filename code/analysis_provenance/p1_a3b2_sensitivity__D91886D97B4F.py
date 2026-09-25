import argparse,csv,json,math,os,re,statistics
from collections import defaultdict

import numpy as np

EPS=1e-12

def F(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None

def training_files(folder):
    out={}
    for fn in sorted(os.listdir(folder)):
        m=re.search(r"_ID_(\d+)\.csv$",fn,re.I)
        if m and int(m.group(1))<=3 and fn.lower().endswith(".csv"):
            out[fn]=os.path.join(folder,fn)
    return out

def load_env(path):
    env={"lon":{},"lat":{}}
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            env[r["axis"]][r["variable"]]=(float(r["common_low"]),float(r["common_high"]))
    return env

def load_rep(path):
    rep={"lon":set(),"lat":set()}
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r["pair_represented"]).lower()=="true":
                rep[r["axis"]].add(r["file_name"])
    return rep

def load_pilot(path,pcfg):
    out={}
    S={cfg:float(pcfg[cfg]["S_wing"]) for cfg in pcfg}
    b={cfg:float(pcfg[cfg]["b"]) for cfg in pcfg}
    et={cfg:float(pcfg[cfg]["eta_prop_T"]) for cfg in pcfg}
    eq={cfg:float(pcfg[cfg]["eta_prop_Q"]) for cfg in pcfg}

    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cfg=r["configuration"];fn=r["file_name"];idx=int(r["sample_index"])
            alpha=F(r.get("alpha"));beta=F(r.get("beta"))
            qbar=F(r.get("qbar"))
            T=F(r.get("T_prop_N"));Q=F(r.get("Q_drag_Nm"))

            y={
                "C_L_lift":F(r.get("C_L_lift")),
                "C_m_pitch":F(r.get("C_m_pitch")),
                "C_Y_wind":F(r.get("C_Y_wind")),
                "C_l_roll":F(r.get("C_l_roll")),
                "C_n_yaw":F(r.get("C_n_yaw"))
            }

            # eta=1 sensitivity branch. Clean eta is exactly 1; iced differs.
            y_eta=dict(y)
            if qbar is not None and qbar>1e-12 and alpha is not None and beta is not None:
                if T is not None and et[cfg]>0:
                    T1=T/et[cfg]
                    dX=-(T1-T)
                    y_eta["C_L_lift"]=y["C_L_lift"] + math.sin(alpha)*dX/(qbar*S[cfg])
                    y_eta["C_Y_wind"]=y["C_Y_wind"] + (-math.cos(alpha)*math.sin(beta))*dX/(qbar*S[cfg])
                if Q is not None and eq[cfg]>0:
                    Q1=Q/eq[cfg]
                    dL=(Q1-Q)
                    y_eta["C_l_roll"]=y["C_l_roll"] + dL/(qbar*S[cfg]*b[cfg])

            out[(cfg,fn,idx)]={"primary":y,"eta1":y_eta}
    return out

def axis_of(fn): return "lon" if fn.lower().startswith("lon_") else "lat"

def feat(row,axis,b,c):
    Va=F(row.get("Va_EKF_CG"));alpha=F(row.get("alpha"));beta=F(row.get("beta"))
    p=F(row.get("p"));q=F(row.get("q"));r=F(row.get("r"))
    de=F(row.get("elevator"));da=F(row.get("aileron"))
    if Va is None or Va<=1e-8:return None
    if axis=="lon":
        x={"Va":Va,"alpha":alpha,"q_hat":None if q is None else q*c/(2*Va),"delta_e":de}
    else:
        x={"Va":Va,"alpha":alpha,"beta":beta,"p_hat":None if p is None else p*b/(2*Va),
           "r_hat":None if r is None else r*b/(2*Va),"delta_a":da}
    if any(v is None or not math.isfinite(v) for v in x.values()):return None
    return x

def inside(x,env):
    return all(env[k][0]<=x[k]<=env[k][1] for k in env)

def demean(rows,predictors,outcome):
    groups=defaultdict(list)
    for r in rows: groups[(r["file_name"],r["configuration"])].append(r)
    out=[]
    sizes={}
    for g,rr in groups.items():
        sizes[g]=len(rr)
        mu={k:sum(r[k] for r in rr)/len(rr) for k in predictors}
        my=sum(r[outcome] for r in rr)/len(rr)
        for r in rr:
            z=dict(r)
            z["_group"]=g
            z["_y"]=r[outcome]-my
            z["_x"]=[r[k]-mu[k] for k in predictors]
            out.append(z)
    return out,sizes

def fit(rows,predictors,outcome,weight_mode="row"):
    dm,sizes=demean(rows,predictors,outcome)
    p=len(predictors)
    X=[];Y=[];W=[];groups=[];sample_index=[]
    for r in dm:
        x=r["_x"]
        vec=x+[0.0]*p if r["configuration"]=="clean" else [0.0]*p+x
        X.append(vec);Y.append(r["_y"]);groups.append(r["_group"]);sample_index.append(r["sample_index"])
        if weight_mode=="equal_flight":
            W.append(1.0/sizes[r["_group"]])
        else:
            W.append(1.0)

    X=np.asarray(X,float);Y=np.asarray(Y,float);W=np.asarray(W,float)
    sw=np.sqrt(W)
    Xw=X*sw[:,None];Yw=Y*sw
    scales=np.sqrt(np.mean(Xw*Xw,axis=0))
    if np.any(scales<=1e-14):raise RuntimeError("zero design scale")
    Xs=Xw/scales
    bs,_,rank,svals=np.linalg.lstsq(Xs,Yw,rcond=None)
    beta=bs/scales
    cond=float(np.max(svals)/np.min(svals))
    resid=Y-X@beta
    return beta,int(rank),cond,resid,groups,sample_index

def bh_holm(pvals):
    m=len(pvals)
    order=sorted(range(m),key=lambda i:pvals[i])

    bh=[None]*m
    running=1.0
    for rank0 in range(m-1,-1,-1):
        idx=order[rank0]
        rank=rank0+1
        val=pvals[idx]*m/rank
        running=min(running,val)
        bh[idx]=min(1.0,running)

    holm=[None]*m
    running=0.0
    for rank0,idx in enumerate(order):
        val=(m-rank0)*pvals[idx]
        running=max(running,val)
        holm[idx]=min(1.0,running)
    return bh,holm

def lag1_corr(vals):
    if len(vals)<3:return None
    a=np.asarray(vals[:-1],float);b=np.asarray(vals[1:],float)
    if np.std(a)<=1e-14 or np.std(b)<=1e-14:return None
    return float(np.corrcoef(a,b)[0,1])

def sign(x):
    if abs(x)<=1e-15:return 0
    return 1 if x>0 else -1

def ratio_abs(a,b):
    if abs(b)<=1e-15:return None
    return abs(a/b)

def main():
    ap=argparse.ArgumentParser()
    for name in ("clean_dir","iced_dir","pilot_samples","envelope","pair_representation",
                 "b1_estimates","clean_json","iced_json","out"):
        ap.add_argument("--"+name.replace("_","-"),required=True)
    a=ap.parse_args()

    pcfg={}
    for cfg,path in [("clean",a.clean_json),("iced",a.iced_json)]:
        with open(path,"r",encoding="utf-8-sig") as f:pcfg[cfg]=json.load(f)
    b=float(pcfg["clean"]["b"]);c=float(pcfg["clean"]["c"])

    env=load_env(a.envelope)
    rep=load_rep(a.pair_representation)
    pilot=load_pilot(a.pilot_samples,pcfg)

    files={"clean":training_files(a.clean_dir),"iced":training_files(a.iced_dir)}

    rows=[]
    for cfg in ("clean","iced"):
        for fn,path in files[cfg].items():
            axis=axis_of(fn)
            if fn not in rep[axis]:continue
            with open(path,"r",newline="",encoding="utf-8-sig") as f:src=list(csv.DictReader(f))
            for idx,r in enumerate(src):
                key=(cfg,fn,idx)
                if key not in pilot:continue
                x=feat(r,axis,b,c)
                if x is None or not inside(x,env[axis]):continue
                y=pilot[key]
                needed=["C_L_lift","C_m_pitch"] if axis=="lon" else ["C_Y_wind","C_l_roll","C_n_yaw"]
                if any(y["primary"][k] is None for k in needed):continue
                rows.append({"configuration":cfg,"file_name":fn,"sample_index":idx,"axis":axis,**x,
                             **y["primary"],**{k+"_eta1":v for k,v in y["eta1"].items()}})

    models=[
        ("lon","C_L_lift",["alpha","q_hat","delta_e"],["Va"]),
        ("lon","C_m_pitch",["alpha","q_hat","delta_e"],["Va"]),
        ("lat","C_Y_wind",["beta","p_hat","r_hat","delta_a"],["Va","alpha"]),
        ("lat","C_l_roll",["beta","p_hat","r_hat","delta_a"],["Va","alpha"]),
        ("lat","C_n_yaw",["beta","p_hat","r_hat","delta_a"],["Va","alpha"])
    ]

    # B1 reference
    b1=[]
    with open(a.b1_estimates,"r",newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            z=dict(r)
            for k in ("clean_estimate","iced_estimate","delta_iced_minus_clean","delta_ci95_low","delta_ci95_high",
                      "delta_p_two_sided","delta_loo_sign_consistency"):
                z[k]=F(z.get(k))
            b1.append(z)

    alt={}
    diagnostics=[]
    residual_rows=[]
    max_primary_repl_err=0.0
    all_alt_integrity=True

    for axis,outcome,mainpred,nuis in models:
        rr=[r for r in rows if r["axis"]==axis]

        # Primary replication
        bp,rankp,condp,residp,groupsp,idxp=fit(rr,mainpred,outcome,"row")
        p=len(mainpred)
        if rankp!=2*p or condp>30:all_alt_integrity=False

        # Equal-flight weighting
        bw,rankw,condw,_,_,_=fit(rr,mainpred,outcome,"equal_flight")
        if rankw!=2*p or condw>30:all_alt_integrity=False

        # Operating-point augmented sensitivity
        augpred=mainpred+nuis
        ba,ranka,conda,_,_,_=fit(rr,augpred,outcome,"row")
        pa=len(augpred)
        if ranka!=2*pa or conda>30:all_alt_integrity=False

        # eta=1 outcome sensitivity on same primary basis
        eta_out=outcome+"_eta1"
        be,ranke,conde,_,_,_=fit(rr,mainpred,eta_out,"row")
        if ranke!=2*p or conde>30:all_alt_integrity=False

        # Residual lag-1 by file/config.
        gr=defaultdict(list)
        for r0,e,g,idx in zip(rr,residp,groupsp,idxp):
            gr[g].append((idx,float(e)))
        rhos=[]
        for g,v in gr.items():
            v=sorted(v)
            rho=lag1_corr([x[1] for x in v])
            if rho is not None:
                rhos.append(rho)
                residual_rows.append({"axis":axis,"outcome":outcome,"configuration":g[1],
                                      "file_name":g[0],"lag1_residual_corr":rho})
        med_rho=statistics.median(rhos) if rhos else None
        max_abs=max(abs(x) for x in rhos) if rhos else None

        diagnostics.append({
            "axis":axis,"outcome":outcome,
            "primary_rank":rankp,"primary_cond":condp,
            "equal_flight_rank":rankw,"equal_flight_cond":condw,
            "augmented_rank":ranka,"augmented_cond":conda,
            "eta1_rank":ranke,"eta1_cond":conde,
            "median_lag1_residual_corr":med_rho,
            "max_abs_lag1_residual_corr":max_abs
        })

        for j,pred in enumerate(mainpred):
            key=(axis,outcome,pred)
            alt[key]={
                "primary_clean":float(bp[j]),
                "primary_iced":float(bp[p+j]),
                "primary_delta":float(bp[p+j]-bp[j]),
                "equal_weight_clean":float(bw[j]),
                "equal_weight_iced":float(bw[p+j]),
                "equal_weight_delta":float(bw[p+j]-bw[j]),
                "augmented_clean":float(ba[j]),
                "augmented_iced":float(ba[pa+j]),
                "augmented_delta":float(ba[pa+j]-ba[j]),
                "eta1_clean":float(be[j]),
                "eta1_iced":float(be[p+j]),
                "eta1_delta":float(be[p+j]-be[j])
            }

    # exact B1 primary replication check
    for r in b1:
        key=(r["axis"],r["outcome"],r["predictor"])
        z=alt[key]
        max_primary_repl_err=max(max_primary_repl_err,
                                 abs(z["primary_clean"]-r["clean_estimate"]),
                                 abs(z["primary_iced"]-r["iced_estimate"]),
                                 abs(z["primary_delta"]-r["delta_iced_minus_clean"]))

    pvals=[r["delta_p_two_sided"] for r in b1]
    if any(p is None for p in pvals):
        raise RuntimeError("B1 p-values missing")
    bh,holm=bh_holm(pvals)

    sensitivity_rows=[]
    strong_count=0
    fdr_count=0
    primary_ci_count=0

    for i,r in enumerate(b1):
        key=(r["axis"],r["outcome"],r["predictor"])
        z=alt[key]
        d=r["delta_iced_minus_clean"]
        ci_excludes=(r["delta_ci95_low"]>0 and r["delta_ci95_high"]>0) or (r["delta_ci95_low"]<0 and r["delta_ci95_high"]<0)
        if ci_excludes:primary_ci_count+=1

        ew_same=sign(z["equal_weight_delta"])==sign(d)
        aug_same=sign(z["augmented_delta"])==sign(d)
        eta_same=sign(z["eta1_delta"])==sign(d)

        ew_ratio=ratio_abs(z["equal_weight_delta"],d)
        aug_ratio=ratio_abs(z["augmented_delta"],d)
        eta_ratio=ratio_abs(z["eta1_delta"],d)

        signcons=r["delta_loo_sign_consistency"]
        base_stable=(
            signcons is not None and signcons>=0.90 and
            ew_same and aug_same and
            ew_ratio is not None and 0.5<=ew_ratio<=1.5 and
            aug_ratio is not None and 0.5<=aug_ratio<=1.5
        )

        # eta sensitivity is relevant only when eta1 actually changes the derivative by >5%.
        eta_material=(eta_ratio is not None and abs(eta_ratio-1.0)>0.05)
        eta_stable=(eta_same and eta_ratio is not None and 0.5<=eta_ratio<=1.5)

        sensitivity_stable=base_stable and (eta_stable if eta_material else True)

        fdr_supported=ci_excludes and bh[i]<=0.05 and sensitivity_stable
        holm_supported=ci_excludes and holm[i]<=0.05 and sensitivity_stable
        if fdr_supported:fdr_count+=1
        if holm_supported:strong_count+=1

        label=("STRONG_HOLM_ROBUST" if holm_supported else
               "FDR_ROBUST" if fdr_supported else
               "PRIMARY_SIGNAL_MODEL_SENSITIVE" if ci_excludes else
               "NO_CLEAR_PRIMARY_DIFFERENCE")

        sensitivity_rows.append({
            "axis":r["axis"],"outcome":r["outcome"],"predictor":r["predictor"],
            "primary_clean":r["clean_estimate"],"primary_iced":r["iced_estimate"],
            "primary_delta":d,"primary_ci95_low":r["delta_ci95_low"],"primary_ci95_high":r["delta_ci95_high"],
            "primary_p":r["delta_p_two_sided"],"bh_fdr_p":bh[i],"holm_p":holm[i],
            "loo_sign_consistency":signcons,
            "equal_weight_delta":z["equal_weight_delta"],"equal_weight_same_sign":ew_same,"equal_weight_abs_ratio":ew_ratio,
            "augmented_delta":z["augmented_delta"],"augmented_same_sign":aug_same,"augmented_abs_ratio":aug_ratio,
            "eta1_delta":z["eta1_delta"],"eta1_same_sign":eta_same,"eta1_abs_ratio":eta_ratio,"eta1_material":eta_material,
            "sensitivity_stable":sensitivity_stable,
            "classification":label
        })

    # Gate is methodological integrity, not number of "significant" findings.
    global_pass=(max_primary_repl_err<1e-10 and all_alt_integrity and len(rows)>0)

    with open(os.path.join(a.out,"P1_A3B2_EFFECT_ROBUSTNESS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=list(sensitivity_rows[0].keys())
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sensitivity_rows)

    with open(os.path.join(a.out,"P1_A3B2_MODEL_ADEQUACY.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=list(diagnostics[0].keys())
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(diagnostics)

    with open(os.path.join(a.out,"P1_A3B2_RESIDUAL_AUTOCORRELATION.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=["axis","outcome","configuration","file_name","lag1_residual_corr"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(residual_rows)

    audit={
        "scope":{"validation_files_used":0,"identification_only":True,"rows_used":len(rows)},
        "primary_replication_max_abs_error":max_primary_repl_err,
        "alternative_model_integrity_pass":all_alt_integrity,
        "multiplicity":{"tests":len(b1),"primary_ci_excluding_zero":primary_ci_count,
                        "bh_fdr_0p05_and_sensitivity_robust":fdr_count,
                        "holm_0p05_and_sensitivity_robust":strong_count},
        "robustness_rule":{
            "loo_sign_consistency_minimum":0.90,
            "equal_flight_weighting_same_sign":True,
            "operating_point_augmented_same_sign":True,
            "alternative_to_primary_abs_magnitude_ratio_range":[0.5,1.5],
            "eta1_materiality_threshold_fraction":0.05,
            "eta1_if_material_requires_same_sign_and_ratio_range":[0.5,1.5],
            "strong_label":"Holm-adjusted p<=0.05 + primary CI excludes zero + sensitivity robust",
            "fdr_label":"BH-adjusted p<=0.05 + primary CI excludes zero + sensitivity robust"
        },
        "global_pass":global_pass
    }
    with open(os.path.join(a.out,"P1_A3B2_AUDIT.json"),"w",encoding="utf-8") as f:json.dump(audit,f,indent=2)

    print("P1_A3B2_MODEL_ADEQUACY_SUMMARY")
    print("validation_files_used             = 0")
    print("rows_used                         =",len(rows))
    print("primary_replication_max_abs_error =",max_primary_repl_err)
    print("alternative_model_integrity_pass =",all_alt_integrity)
    print("primary_CI_excluding_zero         =",primary_ci_count)
    print("BH_FDR_robust_effects             =",fdr_count)
    print("Holm_strong_robust_effects        =",strong_count)
    print("global_adequacy_gate              =",global_pass)
    print("")
    print("EFFECT ROBUSTNESS CLASSIFICATION")
    for r in sensitivity_rows:
        if r["classification"]!="NO_CLEAR_PRIMARY_DIFFERENCE":
            print("  {0:3s} {1:10s}/{2:8s}: {3}; delta={4:+.6g}, BH={5:.4g}, Holm={6:.4g}, EWratio={7:.3f}, AUGratio={8:.3f}, ETA1ratio={9:.3f}".format(
                r["axis"],r["outcome"],r["predictor"],r["classification"],r["primary_delta"],
                r["bh_fdr_p"],r["holm_p"],r["equal_weight_abs_ratio"],r["augmented_abs_ratio"],r["eta1_abs_ratio"]
            ))
    print("")
    print("RESIDUAL SERIAL CORRELATION")
    for d in diagnostics:
        print("  {0:3s} {1:10s}: median rho1={2:.3f}, max|rho1|={3:.3f}".format(
            d["axis"],d["outcome"],d["median_lag1_residual_corr"],d["max_abs_lag1_residual_corr"]
        ))

if __name__=="__main__":
    main()
