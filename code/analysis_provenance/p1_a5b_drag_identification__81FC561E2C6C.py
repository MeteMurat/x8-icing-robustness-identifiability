import argparse,csv,json,math,os,re
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

EPS=1e-14
PI=math.pi

def F(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None

def load_json(path):
    with open(path,"r",encoding="utf-8-sig") as f:
        return json.load(f)

def load_env(path):
    env={}
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["axis"]=="lon":
                env[r["variable"]]=(float(r["common_low"]),float(r["common_high"]))
    need={"Va","alpha","q_hat","delta_e"}
    if not need.issubset(env):
        raise RuntimeError("Frozen longitudinal envelope incomplete.")
    return env

def load_pre(path):
    with open(path,"r",encoding="utf-8-sig") as f:
        return json.load(f)

def solve3(A,b):
    M=[list(map(float,A[i]))+[float(b[i])] for i in range(3)]
    for k in range(3):
        piv=max(range(k,3),key=lambda r:abs(M[r][k]))
        if abs(M[piv][k])<1e-18:
            raise ArithmeticError("singular 3x3 normal matrix")
        if piv!=k:
            M[k],M[piv]=M[piv],M[k]
        d=M[k][k]
        for j in range(k,4):
            M[k][j]/=d
        for r in range(3):
            if r==k:continue
            q=M[r][k]
            for j in range(k,4):
                M[r][j]-=q*M[k][j]
    return [M[i][3] for i in range(3)]

def local_quad_derivative(t,y,i,half):
    lo=i-half
    hi=i+half+1
    if lo<0 or hi>len(t):
        return None

    tc=t[i]
    xs=[]
    ys=[]
    for k in range(lo,hi):
        if t[k] is None or y[k] is None:
            return None
        xs.append(t[k]-tc)
        ys.append(y[k])

    s0=len(xs)
    s1=sum(xs)
    s2=sum(x*x for x in xs)
    s3=sum(x*x*x for x in xs)
    s4=sum(x*x*x*x for x in xs)

    b0=sum(ys)
    b1=sum(x*v for x,v in zip(xs,ys))
    b2=sum(x*x*v for x,v in zip(xs,ys))

    a=solve3(
        [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]],
        [b0,b1,b2]
    )
    return a[1]

def cross(a,b):
    return (
        a[1]*b[2]-a[2]*b[1],
        a[2]*b[0]-a[0]*b[2],
        a[0]*b[1]-a[1]*b[0]
    )

def add(a,b):
    return tuple(a[i]+b[i] for i in range(3))

def sub(a,b):
    return tuple(a[i]-b[i] for i in range(3))

def scale(s,a):
    return tuple(s*x for x in a)

def poly(c,J):
    return c[0]+c[1]*J+c[2]*J*J+c[3]*J*J*J

def inside(x,env):
    return all(env[k][0] <= x[k] <= env[k][1] for k in env)

def id_files(folder):
    out={}
    for name in sorted(os.listdir(folder)):
        if not name.lower().startswith("lon_") or not name.lower().endswith(".csv"):
            continue
        m=re.search(r"_ID_(\d+)\.csv$",name,re.I)
        if m and int(m.group(1))<=3:
            out[name]=os.path.join(folder,name)
    return out

def reconstruct(path,cfg,pms,env,half,eta_unity=False):
    m=float(pms["mass"])
    g=float(pms["gravity"])
    rho=float(pms["rho"])
    D=float(pms["D"])
    S=float(pms["S_wing"])
    b=float(pms["b"])
    cbar=float(pms["c"])

    etaT=1.0 if eta_unity else float(pms["eta_prop_T"])
    etaQ=1.0 if eta_unity else float(pms["eta_prop_Q"])

    ct=[float(pms["C_T_"+str(k)]) for k in range(4)]
    cq=[float(pms["C_Q_"+str(k)]) for k in range(4)]

    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))

    def col(name):
        return [F(r.get(name)) for r in rows]

    t=col("TIME")
    u=col("u_CG")
    v=col("v_CG")
    w=col("w_CG")
    p=col("p")
    q=col("q")
    rrate=col("r")
    Va=col("Va_EKF_CG")
    alpha=col("alpha")
    beta=col("beta")
    phi=col("phi")
    theta=col("theta")
    Om=col("RPS")
    elevator=col("elevator")

    kept=[]
    invalid=0
    source_cd_sq=0.0
    source_cd_den=0.0
    source_cd_n=0
    prop_sq=0.0
    prop_den=0.0
    prop_n=0

    for i in range(half,len(rows)-half):
        req=[
            t[i],u[i],v[i],w[i],
            p[i],q[i],rrate[i],
            Va[i],alpha[i],beta[i],
            phi[i],theta[i],Om[i],elevator[i]
        ]
        if any(x is None for x in req):
            invalid+=1
            continue

        try:
            du=local_quad_derivative(t,u,i,half)
            dv=local_quad_derivative(t,v,i,half)
            dw=local_quad_derivative(t,w,i,half)
        except Exception:
            invalid+=1
            continue

        if None in (du,dv,dw) or Va[i]<=1e-8:
            invalid+=1
            continue

        omega=(p[i],q[i],rrate[i])
        vel=(u[i],v[i],w[i])
        ain=add((du,dv,dw),cross(omega,vel))

        gb=(
            -g*math.sin(theta[i]),
            g*math.sin(phi[i])*math.cos(theta[i]),
            g*math.cos(phi[i])*math.cos(theta[i])
        )

        n=Om[i]/(2.0*PI)
        if abs(n)<1e-10:
            invalid+=1
            continue

        Jadv=Va[i]/(n*D)
        CT=poly(ct,Jadv)
        CQ=poly(cq,Jadv)

        T=etaT*CT*rho*n*n*(D**4)
        Q=etaQ*CQ*rho*n*n*(D**5)

        Fa=sub(
            sub(scale(m,ain),(T,0.0,0.0)),
            scale(m,gb)
        )

        qbar=0.5*rho*Va[i]*Va[i]
        if qbar<=1e-8:
            invalid+=1
            continue

        ca=math.cos(alpha[i])
        sa=math.sin(alpha[i])
        cb=math.cos(beta[i])
        sb=math.sin(beta[i])

        # Unit vector of relative wind direction in body axes:
        # e_V = [cos a cos b, sin b, sin a cos b]
        # Drag is positive opposite to aerodynamic force projected along e_V.
        Drag=-(Fa[0]*ca*cb + Fa[1]*sb + Fa[2]*sa*cb)
        CD=Drag/(qbar*S)

        qhat=q[i]*cbar/(2.0*Va[i])

        x={
            "Va":Va[i],
            "alpha":alpha[i],
            "q_hat":qhat,
            "delta_e":elevator[i]
        }

        if not inside(x,env):
            continue

        src_cd=F(rows[i].get("CD"))
        if src_cd is not None:
            source_cd_sq+=(CD-src_cd)**2
            source_cd_den+=src_cd*src_cd
            source_cd_n+=1

        src_t=F(rows[i].get("prop_thrust_bodyX"))
        if src_t is not None:
            prop_sq+=(T-src_t)**2
            prop_den+=src_t*src_t
            prop_n+=1

        kept.append({
            "configuration":cfg,
            "file_name":os.path.basename(path),
            "sample_index":i,
            "Va":Va[i],
            "alpha":alpha[i],
            "beta":beta[i],
            "q_hat":qhat,
            "delta_e":elevator[i],
            "C_D":CD,
            "T_prop":T,
            "Q_prop":Q
        })

    src_cd_nrmse=None
    if source_cd_n and source_cd_den>EPS:
        src_cd_nrmse=math.sqrt(source_cd_sq/source_cd_n)/math.sqrt(source_cd_den/source_cd_n)

    prop_nrmse=None
    if prop_n and prop_den>EPS:
        prop_nrmse=math.sqrt(prop_sq/prop_n)/math.sqrt(prop_den/prop_n)

    return {
        "source_rows":len(rows),
        "common_rows":len(kept),
        "invalid_rows":invalid,
        "source_cd_nrmse":src_cd_nrmse,
        "prop_ref_nrmse":prop_nrmse,
        "rows":kept
    }

def pair_demean(rows,cols,ycol,weight_mode):
    groups=defaultdict(list)
    for r in rows:
        groups[r["file_name"]].append(r)

    X=[]
    Y=[]
    W=[]
    cluster=[]

    for pair,rr in groups.items():
        mx={c:sum(r[c] for r in rr)/len(rr) for c in cols}
        my=sum(r[ycol] for r in rr)/len(rr)
        pair_weight=1.0/len(rr) if weight_mode=="equal_pair" else 1.0

        for r in rr:
            X.append([r[c]-mx[c] for c in cols])
            Y.append(r[ycol]-my)
            W.append(pair_weight)
            cluster.append(pair)

    return np.asarray(X,float),np.asarray(Y,float),np.asarray(W,float),np.asarray(cluster,dtype=object)

def build_rows(raw,ref,beta_aug=False):
    out=[]
    for r in raw:
        Va_c=r["Va"]-ref["Va"]
        a_c=r["alpha"]-ref["alpha"]
        q_c=r["q_hat"]-ref["q_hat"]
        de_c=r["delta_e"]-ref["delta_e"]
        ice=1.0 if r["configuration"]=="iced" else 0.0

        z=dict(r)
        z["Va_c"]=Va_c
        z["alpha_c"]=a_c
        z["alpha_c2"]=a_c*a_c
        z["q_hat_c"]=q_c
        z["delta_e_c"]=de_c
        z["ice"]=ice
        z["ice_Va_c"]=ice*Va_c
        z["ice_alpha_c"]=ice*a_c
        z["ice_alpha_c2"]=ice*a_c*a_c
        z["ice_q_hat_c"]=ice*q_c
        z["ice_delta_e_c"]=ice*de_c

        if beta_aug:
            z["beta_c"]=r["beta"]  # frozen physical reference beta=0
        out.append(z)
    return out

def fit_model(raw,ref,weight_mode="row",beta_aug=False):
    rows=build_rows(raw,ref,beta_aug)

    base=[
        "Va_c","alpha_c","alpha_c2","q_hat_c","delta_e_c"
    ]
    cols=base+[
        "ice",
        "ice_Va_c","ice_alpha_c","ice_alpha_c2","ice_q_hat_c","ice_delta_e_c"
    ]

    if beta_aug:
        # Nuisance beta term is configuration-common by pre-analysis contract.
        cols=["beta_c"]+cols

    X,Y,W,clusters=pair_demean(rows,cols,"C_D",weight_mode)

    sw=np.sqrt(W)
    Xw=X*sw[:,None]
    Yw=Y*sw

    scales=np.sqrt(np.mean(Xw*Xw,axis=0))
    if np.any(scales<=1e-14):
        raise RuntimeError("zero design scale")

    Xs=Xw/scales
    bs,_,rank,svals=np.linalg.lstsq(Xs,Yw,rcond=None)
    beta=bs/scales
    cond=float(np.max(svals)/np.min(svals))

    gamma_idx=cols.index("ice")
    gamma0=float(beta[gamma_idx])

    unique=sorted(set(clusters.tolist()))
    loo=[]
    failures=[]

    for g in unique:
        keep=clusters!=g
        Xg=X[keep,:]
        Yg=Y[keep]
        Wg=W[keep]

        swg=np.sqrt(Wg)
        Xgw=Xg*swg[:,None]
        Ygw=Yg*swg
        sg=np.sqrt(np.mean(Xgw*Xgw,axis=0))

        if np.any(sg<=1e-14):
            failures.append(g)
            continue

        try:
            bg,_,rg,svg=np.linalg.lstsq(Xgw/sg,Ygw,rcond=None)
            if int(rg)!=len(cols):
                failures.append(g)
                continue
            bgo=bg/sg
            loo.append((g,float(bgo[gamma_idx])))
        except Exception:
            failures.append(g)

    if len(loo)>=2:
        vals=np.asarray([x[1] for x in loo],float)
        bar=float(np.mean(vals))
        se=math.sqrt((len(vals)-1.0)/len(vals)*float(np.sum((vals-bar)**2)))
        lo=float(np.min(vals))
        hi=float(np.max(vals))
    else:
        se=None
        lo=None
        hi=None

    # Reconstruct pair fixed effects on the original, non-demeaned model for
    # a descriptive clean-reference C_D level.
    pred_no_pair=[]
    for r in rows:
        pred=sum(beta[j]*r[cols[j]] for j in range(len(cols)))
        pred_no_pair.append((r["file_name"],r["configuration"],r["C_D"],pred,r))

    by_pair=defaultdict(list)
    for item in pred_no_pair:
        by_pair[item[0]].append(item)

    pair_intercepts={}
    for pair,items in by_pair.items():
        pair_intercepts[pair]=sum(y-pred for _,_,y,pred,_ in items)/len(items)

    mean_clean_ref=float(np.mean(list(pair_intercepts.values())))
    relative_ref_increment=(gamma0/mean_clean_ref) if mean_clean_ref>EPS else None

    # Average model-implied ice effect across observed frozen-envelope X.
    ice_effects=[]
    for r in rows:
        eff=gamma0
        for nm in ("Va_c","alpha_c","alpha_c2","q_hat_c","delta_e_c"):
            idx=cols.index("ice_"+nm)
            eff+=beta[idx]*r[nm]
        ice_effects.append(eff)
    avg_effect=float(np.mean(ice_effects))

    pair_sizes=defaultdict(int)
    for c in clusters:
        pair_sizes[str(c)]+=1
    max_cluster=max(pair_sizes.values())/len(clusters)

    coef={cols[j]:float(beta[j]) for j in range(len(cols))}

    return {
        "columns":cols,
        "coefficients":coef,
        "rank":int(rank),
        "required_rank":len(cols),
        "condition_number":cond,
        "pairs":len(unique),
        "jackknife_refits":len(loo),
        "jackknife_failures":failures,
        "largest_pair_fraction":max_cluster,
        "gamma0":gamma0,
        "gamma0_se":se,
        "gamma0_loo_min":lo,
        "gamma0_loo_max":hi,
        "mean_clean_reference_CD":mean_clean_ref,
        "relative_reference_increment":relative_ref_increment,
        "average_model_implied_delta_CD":avg_effect
    }

def one_sided_positive_p(est,se,df):
    if se is None or se<=EPS:
        return 0.0 if est>0 else 1.0
    return float(student_t.sf(est/se,df))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-dir",required=True)
    ap.add_argument("--iced-dir",required=True)
    ap.add_argument("--clean-json",required=True)
    ap.add_argument("--iced-json",required=True)
    ap.add_argument("--envelope",required=True)
    ap.add_argument("--preanalysis",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    pms={"clean":load_json(a.clean_json),"iced":load_json(a.iced_json)}
    env=load_env(a.envelope)
    pre=load_pre(a.preanalysis)

    ref={
        "Va":float(pre["frozen_reference_operating_point"]["Va"]),
        "alpha":float(pre["frozen_reference_operating_point"]["alpha"]),
        "q_hat":float(pre["frozen_reference_operating_point"]["q_hat"]),
        "delta_e":float(pre["frozen_reference_operating_point"]["delta_e"])
    }

    files={"clean":id_files(a.clean_dir),"iced":id_files(a.iced_dir)}
    exact=sorted(set(files["clean"]).intersection(files["iced"]))

    if len(exact)!=12:
        raise RuntimeError("Expected 12 exact longitudinal identification pairs; observed %d"%len(exact))

    branch_spec={
        "PRIMARY_W5":{"half":2,"eta_unity":False},
        "ETA1_W5":{"half":2,"eta_unity":True},
        "W7":{"half":3,"eta_unity":False},
        "W9_STRESS":{"half":4,"eta_unity":False}
    }

    branch_rows={}
    branch_support={}
    diagnostics=[]

    # Reconstruct each frozen derivative/propulsion branch once.
    for bname,spec in branch_spec.items():
        allpair={}
        for fn in exact:
            rr={}
            for cfg in ("clean","iced"):
                res=reconstruct(
                    files[cfg][fn],
                    cfg,
                    pms[cfg],
                    env,
                    spec["half"],
                    spec["eta_unity"]
                )
                rr[cfg]=res
                diagnostics.append({
                    "branch":bname,
                    "configuration":cfg,
                    "file_name":fn,
                    "source_rows":res["source_rows"],
                    "common_envelope_rows":res["common_rows"],
                    "invalid_rows":res["invalid_rows"],
                    "source_CD_diagnostic_nrmse":res["source_cd_nrmse"],
                    "prop_thrust_reference_nrmse":res["prop_ref_nrmse"]
                })
            allpair[fn]=rr

        branch_support[bname]=allpair

    # Primary support gate determines canonical represented pair set.
    represented=[]
    support_rows=[]
    for fn in exact:
        c=branch_support["PRIMARY_W5"][fn]["clean"]["common_rows"]
        i=branch_support["PRIMARY_W5"][fn]["iced"]["common_rows"]
        ok=(c>=25 and i>=25)
        support_rows.append({
            "file_name":fn,
            "clean_common_envelope_rows":c,
            "iced_common_envelope_rows":i,
            "represented_primary":ok
        })
        if ok:
            represented.append(fn)

    if len(represented)<10:
        primary_support_pass=False
    else:
        primary_support_pass=True

    # All sensitivity branches use the SAME primary represented pair names.
    for bname in branch_spec:
        rr=[]
        for fn in represented:
            rr.extend(branch_support[bname][fn]["clean"]["rows"])
            rr.extend(branch_support[bname][fn]["iced"]["rows"])
        branch_rows[bname]=rr

    fits={}

    if primary_support_pass:
        fits["PRIMARY_W5"]=fit_model(
            branch_rows["PRIMARY_W5"],ref,
            weight_mode="row",beta_aug=False
        )
        fits["EQUAL_PAIR_W5"]=fit_model(
            branch_rows["PRIMARY_W5"],ref,
            weight_mode="equal_pair",beta_aug=False
        )
        fits["BETA_AUG_W5"]=fit_model(
            branch_rows["PRIMARY_W5"],ref,
            weight_mode="row",beta_aug=True
        )
        fits["ETA1_W5"]=fit_model(
            branch_rows["ETA1_W5"],ref,
            weight_mode="row",beta_aug=False
        )
        fits["W7"]=fit_model(
            branch_rows["W7"],ref,
            weight_mode="row",beta_aug=False
        )
        fits["W9_STRESS"]=fit_model(
            branch_rows["W9_STRESS"],ref,
            weight_mode="row",beta_aug=False
        )
    else:
        fits={}

    # Numerical gates frozen by A5A.
    required_fit_names=[
        "PRIMARY_W5","EQUAL_PAIR_W5","BETA_AUG_W5","ETA1_W5","W7"
    ]

    numerical_pass=primary_support_pass
    if primary_support_pass:
        for name in required_fit_names:
            z=fits[name]
            numerical_pass=numerical_pass and (
                z["rank"]==z["required_rank"] and
                z["condition_number"]<=30.0 and
                z["pairs"]>=10 and
                z["largest_pair_fraction"]<=0.20 and
                z["jackknife_refits"]==z["pairs"]
            )

    if primary_support_pass:
        primary=fits["PRIMARY_W5"]
        raw_p=one_sided_positive_p(
            primary["gamma0"],
            primary["gamma0_se"],
            primary["pairs"]-1
        )

        required_positive={
            "PRIMARY_W5":fits["PRIMARY_W5"]["gamma0"]>0,
            "ETA1_W5":fits["ETA1_W5"]["gamma0"]>0,
            "EQUAL_PAIR_W5":fits["EQUAL_PAIR_W5"]["gamma0"]>0,
            "BETA_AUG_W5":fits["BETA_AUG_W5"]["gamma0"]>0,
            "W7":fits["W7"]["gamma0"]>0
        }

        all_required_positive=all(required_positive.values())

        if primary["gamma0"]<=0:
            classification="NO_POSITIVE_REFERENCE_DRAG_INCREMENT"
        elif not all_required_positive:
            classification="DRAG_SIGNAL_MODEL_OR_PROPULSION_SENSITIVE"
        else:
            classification="DRAG_SIGNAL_ROBUST_IDENTIFICATION"

        hd1_significant=(primary["gamma0"]>0 and raw_p<=0.05)
    else:
        primary=None
        raw_p=None
        required_positive={}
        all_required_positive=False
        classification="INSUFFICIENT_IDENTIFICATION_SUPPORT"
        hd1_significant=False

    execution_pass=(
        len(exact)==12 and
        primary_support_pass and
        numerical_pass
    )

    with open(
        os.path.join(a.out,"P1_A5B_PAIR_SUPPORT.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=[
            "file_name",
            "clean_common_envelope_rows",
            "iced_common_envelope_rows",
            "represented_primary"
        ]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(support_rows)

    with open(
        os.path.join(a.out,"P1_A5B_RECONSTRUCTION_DIAGNOSTICS.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=[
            "branch","configuration","file_name",
            "source_rows","common_envelope_rows","invalid_rows",
            "source_CD_diagnostic_nrmse",
            "prop_thrust_reference_nrmse"
        ]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(diagnostics)

    fit_rows=[]
    for name,z in fits.items():
        fit_rows.append({
            "branch":name,
            "pairs":z["pairs"],
            "rank":z["rank"],
            "required_rank":z["required_rank"],
            "condition_number":z["condition_number"],
            "largest_pair_fraction":z["largest_pair_fraction"],
            "jackknife_refits":z["jackknife_refits"],
            "gamma0_delta_CD_ref":z["gamma0"],
            "gamma0_jackknife_se":z["gamma0_se"],
            "gamma0_loo_min":z["gamma0_loo_min"],
            "gamma0_loo_max":z["gamma0_loo_max"],
            "mean_clean_reference_CD":z["mean_clean_reference_CD"],
            "relative_reference_increment":z["relative_reference_increment"],
            "average_model_implied_delta_CD":z["average_model_implied_delta_CD"]
        })

    with open(
        os.path.join(a.out,"P1_A5B_DRAG_BRANCH_RESULTS.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=list(fit_rows[0].keys()) if fit_rows else [
            "branch","pairs","rank","required_rank","condition_number",
            "largest_pair_fraction","jackknife_refits",
            "gamma0_delta_CD_ref","gamma0_jackknife_se",
            "gamma0_loo_min","gamma0_loo_max",
            "mean_clean_reference_CD","relative_reference_increment",
            "average_model_implied_delta_CD"
        ]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(fit_rows)

    coef_rows=[]
    for name,z in fits.items():
        for k,v in z["coefficients"].items():
            coef_rows.append({
                "branch":name,
                "coefficient":k,
                "estimate":v
            })

    with open(
        os.path.join(a.out,"P1_A5B_DRAG_COEFFICIENTS.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=["branch","coefficient","estimate"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(coef_rows)

    audit={
        "scope":{
            "identification_only":True,
            "validation_drag_outcomes_opened":False,
            "exact_longitudinal_pairs":len(exact),
            "represented_primary_pairs":len(represented)
        },
        "support":{
            "primary_support_pass":primary_support_pass,
            "minimum_pairs":10,
            "minimum_rows_each_configuration_per_pair":25
        },
        "integrity":{
            "required_fit_numerical_pass":numerical_pass
        },
        "primary_hypothesis":{
            "id":"HD1",
            "alternative":"Delta C_D_ref > 0",
            "gamma0":None if primary is None else primary["gamma0"],
            "jackknife_se":None if primary is None else primary["gamma0_se"],
            "one_sided_p":raw_p,
            "significant_at_0p05":hd1_significant
        },
        "required_directional_sensitivities":required_positive,
        "all_required_directional_sensitivities_positive":all_required_positive,
        "classification":classification,
        "W9_is_descriptive_only":True,
        "execution_pass":execution_pass
    }

    with open(
        os.path.join(a.out,"P1_A5B_AUDIT.json"),
        "w",encoding="utf-8"
    ) as f:
        json.dump(audit,f,indent=2)

    print("P1_A5B_IDENTIFICATION_DRAG_SUMMARY")
    print("validation_drag_outcomes_opened    = False")
    print("exact_longitudinal_pairs           =",len(exact))
    print("represented_primary_pairs          =",len(represented))
    print("primary_support_pass               =",primary_support_pass)
    print("required_fit_numerical_pass        =",numerical_pass)
    print("execution_pass                     =",execution_pass)
    print("classification                     =",classification)

    if primary is not None:
        print("")
        print("PRIMARY HD1")
        print("  Delta C_D_ref gamma0             = {:+.9g}".format(primary["gamma0"]))
        print("  jackknife SE                     = {:.9g}".format(primary["gamma0_se"]))
        print("  one-sided p                      = {:.9g}".format(raw_p))
        print("  HD1 p<=0.05                      =",hd1_significant)
        print("  mean clean reference C_D         = {:+.9g}".format(primary["mean_clean_reference_CD"]))
        if primary["relative_reference_increment"] is not None:
            print("  relative reference drag increase = {:+.3%}".format(primary["relative_reference_increment"]))
        print("  average model Delta C_D          = {:+.9g}".format(primary["average_model_implied_delta_CD"]))
        print("")
        print("FROZEN SENSITIVITY BRANCHES")
        for name in (
            "PRIMARY_W5","ETA1_W5","EQUAL_PAIR_W5",
            "BETA_AUG_W5","W7","W9_STRESS"
        ):
            z=fits[name]
            gate="DESCRIPTIVE_ONLY" if name=="W9_STRESS" else (
                "POSITIVE" if z["gamma0"]>0 else "NONPOSITIVE"
            )
            print(
                "  {0:14s}: gamma0={1:+.9g}, cond={2:.3f}, gate={3}".format(
                    name,z["gamma0"],z["condition_number"],gate
                )
            )

if __name__=="__main__":
    main()
