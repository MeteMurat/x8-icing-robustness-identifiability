import argparse,csv,json,math,os,re,statistics
from collections import defaultdict

try:
    import numpy as np
except Exception as e:
    raise SystemExit("NumPy is required for P1-A3B1: %r" % (e,))

try:
    from scipy.stats import t as student_t
    HAVE_SCIPY=True
except Exception:
    HAVE_SCIPY=False

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
        if not fn.lower().endswith(".csv"):continue
        m=re.search(r"_ID_(\d+)\.csv$",fn,re.I)
        if m and int(m.group(1))<=3:
            out[fn]=os.path.join(folder,fn)
    return out

def load_outcomes(path):
    d={}
    with open(path,"r",newline="",encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            key=(r["configuration"],r["file_name"],int(r["sample_index"]))
            d[key]={
                "C_L_lift":F(r.get("C_L_lift")),
                "C_m_pitch":F(r.get("C_m_pitch")),
                "C_Y_wind":F(r.get("C_Y_wind")),
                "C_l_roll":F(r.get("C_l_roll")),
                "C_n_yaw":F(r.get("C_n_yaw"))
            }
    return d

def load_envelope(path):
    env={"lon":{},"lat":{}}
    with open(path,"r",newline="",encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            env[r["axis"]][r["variable"]]=(float(r["common_low"]),float(r["common_high"]))
    return env

def load_represented_pairs(path):
    out={"lon":set(),"lat":set()}
    expected_counts={"lon":{"clean":0,"iced":0},"lat":{"clean":0,"iced":0}}
    with open(path,"r",newline="",encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            flag=str(r["pair_represented"]).strip().lower()=="true"
            if not flag:continue
            axis=r["axis"];fn=r["file_name"]
            out[axis].add(fn)
            expected_counts[axis]["clean"]+=int(r["clean_common_envelope"])
            expected_counts[axis]["iced"]+=int(r["iced_common_envelope"])
    return out,expected_counts

def axis_of(fn):
    return "lon" if fn.lower().startswith("lon_") else "lat"

def features(row,axis,b,c):
    Va=F(row.get("Va_EKF_CG"))
    alpha=F(row.get("alpha"))
    beta=F(row.get("beta"))
    p=F(row.get("p"));q=F(row.get("q"));r=F(row.get("r"))
    de=F(row.get("elevator"));da=F(row.get("aileron"))

    if Va is None or Va<=1e-8:return None

    if axis=="lon":
        vals={
            "Va":Va,
            "alpha":alpha,
            "q_hat":None if q is None else q*c/(2*Va),
            "delta_e":de
        }
    else:
        vals={
            "Va":Va,
            "alpha":alpha,
            "beta":beta,
            "p_hat":None if p is None else p*b/(2*Va),
            "r_hat":None if r is None else r*b/(2*Va),
            "delta_a":da
        }

    if any(v is None or not math.isfinite(v) for v in vals.values()):
        return None
    return vals

def in_envelope(v,env):
    return all(env[k][0] <= v[k] <= env[k][1] for k in env)

def tcrit95(df):
    if HAVE_SCIPY:
        return float(student_t.ppf(0.975,df))
    fallback={11:2.200985,13:2.160369,14:2.144787}
    return fallback.get(df,1.96)

def pvalue_t(tabs,df):
    if HAVE_SCIPY:
        return float(2.0*student_t.sf(abs(tabs),df))
    return None

def within_demean(rows,predictors,outcome):
    # group is exact file x configuration; removes run-specific level offsets.
    groups=defaultdict(list)
    for r in rows:
        groups[(r["file_name"],r["configuration"])].append(r)

    out=[]
    for g,rr in groups.items():
        mx={k:sum(x[k] for x in rr)/len(rr) for k in predictors}
        my=sum(x[outcome] for x in rr)/len(rr)
        for x in rr:
            z=dict(x)
            for k in predictors:z[k+"_dm"]=x[k]-mx[k]
            z[outcome+"_dm"]=x[outcome]-my
            out.append(z)
    return out

def fit_model(rows,axis,outcome,predictors):
    dm=within_demean(rows,predictors,outcome)
    p=len(predictors)

    X=[];Y=[];clusters=[]
    for r in dm:
        x=[r[k+"_dm"] for k in predictors]
        if r["configuration"]=="clean":
            vec=x+[0.0]*p
        else:
            vec=[0.0]*p+x
        X.append(vec)
        Y.append(r[outcome+"_dm"])
        clusters.append(r["file_name"])

    X=np.asarray(X,dtype=float)
    Y=np.asarray(Y,dtype=float)

    # Column scaling is internal only; final coefficients are transformed back.
    scales=np.sqrt(np.mean(X*X,axis=0))
    if np.any(scales<=1e-14):
        raise RuntimeError("zero-scale design column in %s/%s" % (axis,outcome))

    Xs=X/scales
    beta_s,resid,rank,svals=np.linalg.lstsq(Xs,Y,rcond=None)
    beta=beta_s/scales
    yhat=X@beta
    e=Y-yhat

    svals=np.asarray(svals)
    cond=float(np.max(svals)/np.min(svals)) if len(svals) else float("inf")

    sst=float(np.sum(Y*Y))
    sse=float(np.sum(e*e))
    r2=1.0-sse/sst if sst>EPS else None
    rms=math.sqrt(sse/len(Y))

    unique_clusters=sorted(set(clusters))
    G=len(unique_clusters)
    if G<2:raise RuntimeError("too few clusters")

    # Leave-one-exact-maneuver-pair-out jackknife.
    loo=[]
    failures=[]
    cluster_arr=np.asarray(clusters,dtype=object)

    for g in unique_clusters:
        keep=cluster_arr!=g
        Xg=X[keep,:];Yg=Y[keep]
        sc=np.sqrt(np.mean(Xg*Xg,axis=0))
        if np.any(sc<=1e-14):
            failures.append(g);continue
        Xgs=Xg/sc
        try:
            bs,_,rg,svg=np.linalg.lstsq(Xgs,Yg,rcond=None)
            if int(rg)!=2*p:
                failures.append(g);continue
            loo.append((g,bs/sc))
        except Exception:
            failures.append(g)

    if len(loo)!=G:
        raise RuntimeError("jackknife refit failure in %s/%s: %r" % (axis,outcome,failures))

    B=np.vstack([x[1] for x in loo])
    bbar=np.mean(B,axis=0)
    dif=B-bbar
    Vjk=(G-1.0)/G * (dif.T@dif)
    se=np.sqrt(np.maximum(np.diag(Vjk),0.0))

    rows_out=[]
    tcrit=tcrit95(G-1)

    for j,pred in enumerate(predictors):
        clean=float(beta[j]); iced=float(beta[p+j]); delta=iced-clean

        # Difference jackknife from paired leave-one-file estimates.
        dloo=B[:,p+j]-B[:,j]
        dbar=float(np.mean(dloo))
        dse=math.sqrt((G-1.0)/G * float(np.sum((dloo-dbar)**2)))
        tstat=delta/dse if dse>EPS else (float("inf") if abs(delta)>EPS else 0.0)
        pv=pvalue_t(tstat,G-1)

        c_lo=float(np.min(B[:,j]));c_hi=float(np.max(B[:,j]))
        i_lo=float(np.min(B[:,p+j]));i_hi=float(np.max(B[:,p+j]))
        d_lo=float(np.min(dloo));d_hi=float(np.max(dloo))

        sign_ref=0 if abs(delta)<=EPS else (1 if delta>0 else -1)
        if sign_ref==0:
            sign_fraction=None
        else:
            sign_fraction=float(np.mean(np.sign(dloo)==sign_ref))

        rows_out.append({
            "axis":axis,
            "outcome":outcome,
            "predictor":pred,
            "clean_estimate":clean,
            "iced_estimate":iced,
            "delta_iced_minus_clean":delta,
            "jackknife_se_clean":float(se[j]),
            "jackknife_se_iced":float(se[p+j]),
            "jackknife_se_delta":dse,
            "delta_t_stat":tstat,
            "cluster_df":G-1,
            "delta_p_two_sided":pv,
            "delta_ci95_low":delta-tcrit*dse,
            "delta_ci95_high":delta+tcrit*dse,
            "clean_loo_min":c_lo,
            "clean_loo_max":c_hi,
            "iced_loo_min":i_lo,
            "iced_loo_max":i_hi,
            "delta_loo_min":d_lo,
            "delta_loo_max":d_hi,
            "delta_loo_sign_consistency":sign_fraction
        })

    summary={
        "axis":axis,
        "outcome":outcome,
        "n_rows":len(Y),
        "clusters":G,
        "predictors":p,
        "design_rank":int(rank),
        "required_rank":2*p,
        "scaled_condition_number":cond,
        "within_r2":r2,
        "residual_rms":rms,
        "jackknife_refits":len(loo),
        "scipy_t_distribution_available":HAVE_SCIPY
    }

    # Cluster sizes for influence context.
    sizes=defaultdict(int)
    for g in clusters:sizes[g]+=1
    summary["largest_cluster_fraction"]=max(sizes.values())/len(clusters)

    return rows_out,summary,loo

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-dir",required=True)
    ap.add_argument("--iced-dir",required=True)
    ap.add_argument("--pilot-samples",required=True)
    ap.add_argument("--envelope",required=True)
    ap.add_argument("--pair-representation",required=True)
    ap.add_argument("--clean-json",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    with open(a.clean_json,"r",encoding="utf-8-sig") as fh:
        p=json.load(fh)
    b=float(p["b"]);c=float(p["c"])

    outcomes=load_outcomes(a.pilot_samples)
    env=load_envelope(a.envelope)
    represented,expected_counts=load_represented_pairs(a.pair_representation)

    files={
        "clean":training_files(a.clean_dir),
        "iced":training_files(a.iced_dir)
    }

    rows=[]
    errors=[]

    for cfg in ("clean","iced"):
        for fn,path in files[cfg].items():
            axis=axis_of(fn)
            if fn not in represented[axis]:
                continue
            try:
                with open(path,"r",newline="",encoding="utf-8-sig") as fh:
                    src=list(csv.DictReader(fh))
                for idx,r in enumerate(src):
                    key=(cfg,fn,idx)
                    if key not in outcomes:continue
                    x=features(r,axis,b,c)
                    if x is None or not in_envelope(x,env[axis]):continue
                    y=outcomes[key]
                    needed=["C_L_lift","C_m_pitch"] if axis=="lon" else ["C_Y_wind","C_l_roll","C_n_yaw"]
                    if any(y[k] is None for k in needed):continue
                    rows.append({
                        "configuration":cfg,
                        "file_name":fn,
                        "axis":axis,
                        **x,
                        **{k:y[k] for k in needed}
                    })
            except Exception as e:
                errors.append({"configuration":cfg,"file_name":fn,"error":repr(e)})

    actual_counts={
        "lon":{"clean":sum(1 for r in rows if r["axis"]=="lon" and r["configuration"]=="clean"),
               "iced":sum(1 for r in rows if r["axis"]=="lon" and r["configuration"]=="iced")},
        "lat":{"clean":sum(1 for r in rows if r["axis"]=="lat" and r["configuration"]=="clean"),
               "iced":sum(1 for r in rows if r["axis"]=="lat" and r["configuration"]=="iced")}
    }

    models=[
        ("lon","C_L_lift",["alpha","q_hat","delta_e"]),
        ("lon","C_m_pitch",["alpha","q_hat","delta_e"]),
        ("lat","C_Y_wind",["beta","p_hat","r_hat","delta_a"]),
        ("lat","C_l_roll",["beta","p_hat","r_hat","delta_a"]),
        ("lat","C_n_yaw",["beta","p_hat","r_hat","delta_a"])
    ]

    estimates=[]
    summaries=[]
    loo_rows=[]

    for axis,outcome,predictors in models:
        rr=[r for r in rows if r["axis"]==axis]
        est,summ,loo=fit_model(rr,axis,outcome,predictors)
        estimates.extend(est)
        summaries.append(summ)

        pnum=len(predictors)
        for g,beta in loo:
            row={"axis":axis,"outcome":outcome,"left_out_cluster":g}
            for j,pred in enumerate(predictors):
                row["clean_"+pred]=float(beta[j])
                row["iced_"+pred]=float(beta[pnum+j])
                row["delta_"+pred]=float(beta[pnum+j]-beta[j])
            loo_rows.append(row)

    # Gate is numerical/inferential integrity only; significance is NOT a gate.
    count_match=(actual_counts==expected_counts)
    model_integrity=True
    for s in summaries:
        if s["clusters"]<10:model_integrity=False
        if s["design_rank"]!=s["required_rank"]:model_integrity=False
        if not math.isfinite(s["scaled_condition_number"]) or s["scaled_condition_number"]>30.0:model_integrity=False
        if s["jackknife_refits"]!=s["clusters"]:model_integrity=False
        if s["largest_cluster_fraction"]>0.20:model_integrity=False

    finite_estimates=True
    for r in estimates:
        for k in ("clean_estimate","iced_estimate","delta_iced_minus_clean","jackknife_se_delta",
                  "delta_ci95_low","delta_ci95_high"):
            if r[k] is None or not math.isfinite(float(r[k])):
                finite_estimates=False

    gate=(len(errors)==0 and count_match and model_integrity and finite_estimates)

    with open(os.path.join(a.out,"P1_A3B1_DERIVATIVE_ESTIMATES.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        fields=[
            "axis","outcome","predictor","clean_estimate","iced_estimate","delta_iced_minus_clean",
            "jackknife_se_clean","jackknife_se_iced","jackknife_se_delta","delta_t_stat","cluster_df",
            "delta_p_two_sided","delta_ci95_low","delta_ci95_high",
            "clean_loo_min","clean_loo_max","iced_loo_min","iced_loo_max",
            "delta_loo_min","delta_loo_max","delta_loo_sign_consistency"
        ]
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(estimates)

    with open(os.path.join(a.out,"P1_A3B1_MODEL_SUMMARY.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        fields=[
            "axis","outcome","n_rows","clusters","predictors","design_rank","required_rank",
            "scaled_condition_number","within_r2","residual_rms","jackknife_refits",
            "largest_cluster_fraction","scipy_t_distribution_available"
        ]
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader();w.writerows(summaries)

    # LOO rows have model-dependent columns.
    allkeys=set()
    for r in loo_rows:allkeys.update(r.keys())
    base=["axis","outcome","left_out_cluster"]
    extras=sorted(k for k in allkeys if k not in base)
    with open(os.path.join(a.out,"P1_A3B1_LEAVE_ONE_PAIR_OUT.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=base+extras);w.writeheader();w.writerows(loo_rows)

    with open(os.path.join(a.out,"P1_A3B1_PARSE_ERRORS.csv"),"w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=["configuration","file_name","error"]);w.writeheader();w.writerows(errors)

    audit={
        "scope":{
            "validation_files_used":0,
            "identification_only":True,
            "represented_pair_rule":"use only B0 exact pairs with >=25 common-envelope rows in both clean and iced",
            "represented_pairs":{"lon":len(represented["lon"]),"lat":len(represented["lat"])},
            "expected_rows":expected_counts,
            "actual_rows":actual_counts
        },
        "model":{
            "intercept_policy":"file-by-configuration fixed intercept removed by within-group demeaning",
            "longitudinal_predictors":["alpha","q_hat","delta_e"],
            "longitudinal_outcomes":["C_L_lift","C_m_pitch"],
            "lateral_predictors":["beta","p_hat","r_hat","delta_a"],
            "lateral_outcomes":["C_Y_wind","C_l_roll","C_n_yaw"],
            "point_estimation":"OLS slopes after within-flight demeaning",
            "inference":"leave-one-exact-maneuver-pair-out cluster jackknife; t interval with df=G-1",
            "significance_is_gate":False
        },
        "predeclared_integrity_gate":{
            "minimum_clusters":10,
            "full_rank_required":True,
            "maximum_scaled_condition_number":30.0,
            "maximum_largest_cluster_fraction":0.20,
            "all_cluster_jackknife_refits_required":True,
            "all_estimates_finite":True,
            "row_counts_must_match_frozen_B0_pair_support":True
        },
        "parse_errors":len(errors),
        "row_count_match":count_match,
        "model_integrity_pass":model_integrity,
        "finite_estimates_pass":finite_estimates,
        "global_pass":gate
    }

    with open(os.path.join(a.out,"P1_A3B1_FIT_AUDIT.json"),"w",encoding="utf-8") as fh:
        json.dump(audit,fh,indent=2)

    print("P1_A3B1_DERIVATIVE_ESTIMATION_SUMMARY")
    print("validation_files_used     = 0")
    print("represented_pairs lon/lat =",len(represented["lon"]),"/",len(represented["lat"]))
    print("expected_rows             =",expected_counts)
    print("actual_rows               =",actual_counts)
    print("parse_errors              =",len(errors))
    print("row_count_match           =",count_match)
    print("model_integrity_pass      =",model_integrity)
    print("finite_estimates_pass     =",finite_estimates)
    print("global_fit_gate           =",gate)
    print("")
    print("MODEL SUMMARY")
    for s in summaries:
        print("  {0:3s} {1:10s}: n={2}, G={3}, rank={4}/{5}, cond={6:.3f}, R2w={7:.3f}, clustermax={8:.3f}".format(
            s["axis"],s["outcome"],s["n_rows"],s["clusters"],s["design_rank"],s["required_rank"],
            s["scaled_condition_number"],s["within_r2"],s["largest_cluster_fraction"]
        ))
    print("")
    print("DERIVATIVE DIFFERENCES (ICED - CLEAN; PROVISIONAL UNTIL NEXT ROBUSTNESS STAGE)")
    for r in estimates:
        print("  {0:3s} {1:10s} / {2:8s}: clean={3:+.6g}, iced={4:+.6g}, delta={5:+.6g}, 95%CI=[{6:+.6g},{7:+.6g}], p={8}".format(
            r["axis"],r["outcome"],r["predictor"],r["clean_estimate"],r["iced_estimate"],
            r["delta_iced_minus_clean"],r["delta_ci95_low"],r["delta_ci95_high"],
            "NA" if r["delta_p_two_sided"] is None else "{:.6g}".format(r["delta_p_two_sided"])
        ))

if __name__=="__main__":
    main()
