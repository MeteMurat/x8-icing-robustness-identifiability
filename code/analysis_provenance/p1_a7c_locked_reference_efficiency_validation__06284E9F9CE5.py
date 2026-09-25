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

def solve3(A,b):
    M=[list(map(float,A[i]))+[float(b[i])] for i in range(3)]
    for k in range(3):
        piv=max(range(k,3),key=lambda r:abs(M[r][k]))
        if abs(M[piv][k])<1e-18:
            raise ArithmeticError("singular local polynomial normal matrix")
        if piv!=k:
            M[k],M[piv]=M[piv],M[k]
        d=M[k][k]
        for j in range(k,4):
            M[k][j]/=d
        for r in range(3):
            if r==k:
                continue
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

def val_files(folder):
    out={}
    for name in sorted(os.listdir(folder)):
        if not name.lower().startswith("lon_") or not name.lower().endswith(".csv"):
            continue
        m=re.search(r"_ID_(\d+)\.csv$",name,re.I)
        if m and int(m.group(1))>=4:
            out[name]=os.path.join(folder,name)
    return out

def reconstruct(path,cfg,pms,env,half,eta_unity=False):
    mass=float(pms["mass"])
    grav=float(pms["gravity"])
    rho=float(pms["rho"])
    Dprop=float(pms["D"])
    S=float(pms["S_wing"])
    cbar=float(pms["c"])

    etaT=1.0 if eta_unity else float(pms["eta_prop_T"])
    ct=[float(pms["C_T_"+str(k)]) for k in range(4)]

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
    rr=col("r")
    Va=col("Va_EKF_CG")
    alpha=col("alpha")
    beta=col("beta")
    phi=col("phi")
    theta=col("theta")
    Om=col("RPS")
    elevator=col("elevator")

    kept=[]
    invalid=0
    prop_sq=0.0
    prop_den=0.0
    prop_n=0

    for i in range(half,len(rows)-half):
        req=[
            t[i],u[i],v[i],w[i],
            p[i],q[i],rr[i],
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

        omega=(p[i],q[i],rr[i])
        vel=(u[i],v[i],w[i])
        ain=add((du,dv,dw),cross(omega,vel))

        gbody=(
            -grav*math.sin(theta[i]),
            grav*math.sin(phi[i])*math.cos(theta[i]),
            grav*math.cos(phi[i])*math.cos(theta[i])
        )

        n=Om[i]/(2.0*PI)
        if abs(n)<1e-10:
            invalid+=1
            continue

        Jadv=Va[i]/(n*Dprop)
        CT=poly(ct,Jadv)
        Thrust=etaT*CT*rho*n*n*(Dprop**4)

        Fa=sub(
            sub(scale(mass,ain),(Thrust,0.0,0.0)),
            scale(mass,gbody)
        )

        qbar=0.5*rho*Va[i]*Va[i]
        if qbar<=1e-8:
            invalid+=1
            continue

        ca=math.cos(alpha[i])
        sa=math.sin(alpha[i])
        cb=math.cos(beta[i])
        sb=math.sin(beta[i])

        Drag=-(Fa[0]*ca*cb + Fa[1]*sb + Fa[2]*sa*cb)
        Lift=Fa[0]*sa - Fa[2]*ca

        CD=Drag/(qbar*S)
        CL=Lift/(qbar*S)
        qhat=q[i]*cbar/(2.0*Va[i])

        x={
            "Va":Va[i],
            "alpha":alpha[i],
            "q_hat":qhat,
            "delta_e":elevator[i]
        }

        if not inside(x,env):
            continue

        src_t=F(rows[i].get("prop_thrust_bodyX"))
        if src_t is not None:
            prop_sq+=(Thrust-src_t)**2
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
            "C_L":CL,
            "C_D":CD
        })

    prop_nrmse=None
    if prop_n and prop_den>EPS:
        prop_nrmse=math.sqrt(prop_sq/prop_n)/math.sqrt(prop_den/prop_n)

    return {
        "source_rows":len(rows),
        "common_rows":len(kept),
        "invalid_rows":invalid,
        "prop_thrust_reference_nrmse":prop_nrmse,
        "rows":kept
    }

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
            z["beta_c"]=r["beta"]

        out.append(z)

    return out

def design_columns(beta_aug=False):
    base=["Va_c","alpha_c","alpha_c2","q_hat_c","delta_e_c"]
    cols=base+[
        "ice",
        "ice_Va_c",
        "ice_alpha_c",
        "ice_alpha_c2",
        "ice_q_hat_c",
        "ice_delta_e_c"
    ]
    if beta_aug:
        cols=["beta_c"]+cols
    return cols

def fit_response(rows,ycol,weight_mode="row",beta_aug=False):
    cols=design_columns(beta_aug)

    groups=defaultdict(list)
    for r in rows:
        groups[r["file_name"]].append(r)

    X=[]
    Y=[]
    W=[]
    clusters=[]

    for pair,rr in groups.items():
        mx={c:sum(r[c] for r in rr)/len(rr) for c in cols}
        my=sum(r[ycol] for r in rr)/len(rr)
        pair_weight=1.0/len(rr) if weight_mode=="equal_pair" else 1.0

        for r in rr:
            X.append([r[c]-mx[c] for c in cols])
            Y.append(r[ycol]-my)
            W.append(pair_weight)
            clusters.append(pair)

    X=np.asarray(X,float)
    Y=np.asarray(Y,float)
    W=np.asarray(W,float)
    clusters=np.asarray(clusters,dtype=object)

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

    pair_intercepts={}
    for pair,rr in groups.items():
        offsets=[]
        for r in rr:
            systematic=sum(beta[j]*r[cols[j]] for j in range(len(cols)))
            offsets.append(r[ycol]-systematic)
        pair_intercepts[pair]=float(np.mean(offsets))

    clean_ref=float(np.mean(list(pair_intercepts.values())))
    iced_ref=clean_ref+gamma0

    pair_sizes={pair:len(rr) for pair,rr in groups.items()}
    largest_pair_fraction=max(pair_sizes.values())/sum(pair_sizes.values())

    return {
        "columns":cols,
        "coefficients":{cols[j]:float(beta[j]) for j in range(len(cols))},
        "rank":int(rank),
        "required_rank":len(cols),
        "condition_number":cond,
        "pairs":len(groups),
        "largest_pair_fraction":float(largest_pair_fraction),
        "gamma0":gamma0,
        "clean_ref":clean_ref,
        "iced_ref":iced_ref
    }

def derived(clfit,cdfit):
    CLc=clfit["clean_ref"]
    CLi=clfit["iced_ref"]
    CDc=cdfit["clean_ref"]
    CDi=cdfit["iced_ref"]

    if abs(CDc)<=EPS or abs(CDi)<=EPS:
        Ec=None
        Ei=None
        dE=None
        rel=None
    else:
        Ec=CLc/CDc
        Ei=CLi/CDi
        dE=Ei-Ec
        rel=1.0-Ei/Ec if abs(Ec)>EPS else None

    return {
        "CL_clean_ref":CLc,
        "CL_iced_ref":CLi,
        "delta_CL_ref":CLi-CLc,
        "CD_clean_ref":CDc,
        "CD_iced_ref":CDi,
        "delta_CD_ref":CDi-CDc,
        "E_clean_ref":Ec,
        "E_iced_ref":Ei,
        "delta_E_ref":dE,
        "relative_efficiency_degradation":rel
    }

def jk_se(vals):
    arr=np.asarray(vals,float)
    if len(arr)<2:
        return None
    bar=float(np.mean(arr))
    return math.sqrt((len(arr)-1.0)/len(arr)*float(np.sum((arr-bar)**2)))

def summarize_loo(values):
    out={}
    keys=[
        "CL_clean_ref","CL_iced_ref","delta_CL_ref",
        "CD_clean_ref","CD_iced_ref","delta_CD_ref",
        "E_clean_ref","E_iced_ref","delta_E_ref",
        "relative_efficiency_degradation"
    ]

    for key in keys:
        vals=[z[key] for z in values if z[key] is not None and math.isfinite(z[key])]
        out[key+"_jackknife_se"]=jk_se(vals)
        out[key+"_loo_min"]=float(min(vals)) if vals else None
        out[key+"_loo_max"]=float(max(vals)) if vals else None

    return out

def fit_branch(raw,ref,weight_mode="row",beta_aug=False):
    built=build_rows(raw,ref,beta_aug)

    clfit=fit_response(built,"C_L",weight_mode,beta_aug)
    cdfit=fit_response(built,"C_D",weight_mode,beta_aug)
    full=derived(clfit,cdfit)

    pairs=sorted(set(r["file_name"] for r in built))
    loo=[]
    failures=[]

    for pair in pairs:
        rr=[r for r in raw if r["file_name"]!=pair]
        try:
            bb=build_rows(rr,ref,beta_aug)
            clf=fit_response(bb,"C_L",weight_mode,beta_aug)
            cdf=fit_response(bb,"C_D",weight_mode,beta_aug)

            if clf["rank"]!=clf["required_rank"] or cdf["rank"]!=cdf["required_rank"]:
                failures.append(pair)
                continue

            zz=derived(clf,cdf)
            zz["left_out_pair"]=pair
            loo.append(zz)

        except Exception:
            failures.append(pair)

    loo_summary=summarize_loo(loo)

    full_denominator_positive=(
        full["CD_clean_ref"]>0.0 and
        full["CD_iced_ref"]>0.0
    )

    loo_denominator_positive=(
        len(loo)==len(pairs) and
        all(z["CD_clean_ref"]>0.0 and z["CD_iced_ref"]>0.0 for z in loo)
    )

    denominator_gate=full_denominator_positive and loo_denominator_positive

    rel_se=loo_summary["relative_efficiency_degradation_jackknife_se"]

    if (
        full["relative_efficiency_degradation"] is None or
        rel_se is None or
        rel_se<=EPS
    ):
        p=None
    else:
        p=float(
            student_t.sf(
                full["relative_efficiency_degradation"]/rel_se,
                len(pairs)-1
            )
        )

    out={
        "pairs":len(pairs),
        "jackknife_refits":len(loo),
        "jackknife_failures":failures,

        "CL_rank":clfit["rank"],
        "CL_required_rank":clfit["required_rank"],
        "CL_condition_number":clfit["condition_number"],

        "CD_rank":cdfit["rank"],
        "CD_required_rank":cdfit["required_rank"],
        "CD_condition_number":cdfit["condition_number"],

        "largest_pair_fraction":max(
            clfit["largest_pair_fraction"],
            cdfit["largest_pair_fraction"]
        ),

        "denominator_gate":denominator_gate,
        "full_denominator_positive":full_denominator_positive,
        "loo_denominator_positive":loo_denominator_positive,

        "HE1_one_sided_p":p,
        "CL_coefficients":clfit["coefficients"],
        "CD_coefficients":cdfit["coefficients"]
    }

    out.update(full)
    out.update(loo_summary)
    return out

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

    pms={
        "clean":load_json(a.clean_json),
        "iced":load_json(a.iced_json)
    }

    env=load_env(a.envelope)
    pre=load_json(a.preanalysis)

    frozen=pre["frozen_reference_state"]
    ref={
        "Va":float(frozen["Va_ref_m_s"]),
        "alpha":float(frozen["alpha_ref_rad"]),
        "q_hat":float(frozen["q_hat_ref"]),
        "delta_e":float(frozen["delta_e_ref_rad"])
    }

    files={
        "clean":val_files(a.clean_dir),
        "iced":val_files(a.iced_dir)
    }

    exact=sorted(set(files["clean"]).intersection(files["iced"]))
    unmatched=sorted(set(files["clean"]).symmetric_difference(files["iced"]))

    if len(exact)!=10:
        raise RuntimeError(
            "Expected 10 exact longitudinal validation pairs; observed %d"
            % len(exact)
        )

    branch_spec={
        "PRIMARY_W5":{"half":2,"eta_unity":False},
        "ETA1_W5":{"half":2,"eta_unity":True},
        "W7":{"half":3,"eta_unity":False},
        "W9_STRESS":{"half":4,"eta_unity":False}
    }

    support={}
    diagnostics=[]

    for bname,spec in branch_spec.items():
        bypair={}

        for fn in exact:
            bypair[fn]={}

            for cfg in ("clean","iced"):
                res=reconstruct(
                    files[cfg][fn],
                    cfg,
                    pms[cfg],
                    env,
                    spec["half"],
                    spec["eta_unity"]
                )

                bypair[fn][cfg]=res

                diagnostics.append({
                    "branch":bname,
                    "configuration":cfg,
                    "file_name":fn,
                    "source_rows":res["source_rows"],
                    "common_envelope_rows":res["common_rows"],
                    "invalid_rows":res["invalid_rows"],
                    "prop_thrust_reference_nrmse":res["prop_thrust_reference_nrmse"]
                })

        support[bname]=bypair

    represented=[]
    support_rows=[]

    for fn in exact:
        c=support["PRIMARY_W5"][fn]["clean"]["common_rows"]
        i=support["PRIMARY_W5"][fn]["iced"]["common_rows"]
        ok=(c>=25 and i>=25)

        support_rows.append({
            "file_name":fn,
            "clean_common_envelope_rows":c,
            "iced_common_envelope_rows":i,
            "represented_primary":ok
        })

        if ok:
            represented.append(fn)

    primary_support_pass=len(represented)>=6

    required_physical=["PRIMARY_W5","ETA1_W5","W7"]
    sensitivity_row_support={}

    for bname in required_physical:
        ok=True
        for fn in represented:
            if (
                support[bname][fn]["clean"]["common_rows"]<25 or
                support[bname][fn]["iced"]["common_rows"]<25
            ):
                ok=False
        sensitivity_row_support[bname]=ok

    required_row_support_pass=all(sensitivity_row_support.values())

    branch_rows={}
    for bname in branch_spec:
        rr=[]
        for fn in represented:
            rr.extend(support[bname][fn]["clean"]["rows"])
            rr.extend(support[bname][fn]["iced"]["rows"])
        branch_rows[bname]=rr

    fits={}

    if primary_support_pass and required_row_support_pass:
        fits["PRIMARY_W5"]=fit_branch(
            branch_rows["PRIMARY_W5"],ref,"row",False
        )

        fits["ETA1_W5"]=fit_branch(
            branch_rows["ETA1_W5"],ref,"row",False
        )

        fits["EQUAL_PAIR_W5"]=fit_branch(
            branch_rows["PRIMARY_W5"],ref,"equal_pair",False
        )

        fits["BETA_AUG_W5"]=fit_branch(
            branch_rows["PRIMARY_W5"],ref,"row",True
        )

        fits["W7"]=fit_branch(
            branch_rows["W7"],ref,"row",False
        )

        fits["W9_STRESS"]=fit_branch(
            branch_rows["W9_STRESS"],ref,"row",False
        )

    required=[
        "PRIMARY_W5",
        "ETA1_W5",
        "EQUAL_PAIR_W5",
        "BETA_AUG_W5",
        "W7"
    ]

    numerical_pass=primary_support_pass and required_row_support_pass

    if numerical_pass:
        for name in required:
            z=fits[name]
            numerical_pass=numerical_pass and (
                z["pairs"]>=6 and
                z["jackknife_refits"]==z["pairs"] and
                z["CL_rank"]==z["CL_required_rank"] and
                z["CD_rank"]==z["CD_required_rank"] and
                z["CL_condition_number"]<=30.0 and
                z["CD_condition_number"]<=30.0 and
                z["largest_pair_fraction"]<=0.25
            )

    denominator_pass=numerical_pass
    if numerical_pass:
        denominator_pass=all(fits[name]["denominator_gate"] for name in required)

    if numerical_pass and denominator_pass:
        primary=fits["PRIMARY_W5"]

        required_positive={
            name:(
                fits[name]["relative_efficiency_degradation"] is not None and
                fits[name]["relative_efficiency_degradation"]>0.0
            )
            for name in required
        }

        all_required_positive=all(required_positive.values())
        p=primary["HE1_one_sided_p"]

        if primary["relative_efficiency_degradation"]<=0.0:
            classification="EFFICIENCY_NOT_REPLICATED"
        elif not all_required_positive:
            classification="EFFICIENCY_MODEL_SENSITIVE_VALIDATION"
        elif p is not None and p<=0.05:
            classification="EFFICIENCY_CONFIRMED_STRONG"
        else:
            classification="EFFICIENCY_DIRECTIONALLY_REPLICATED"

    elif numerical_pass and not denominator_pass:
        primary=fits.get("PRIMARY_W5")
        required_positive={}
        all_required_positive=False
        classification="INSUFFICIENT_SUPPORT_DENOMINATOR_GATE"

    else:
        primary=fits.get("PRIMARY_W5")
        required_positive={}
        all_required_positive=False
        classification="INSUFFICIENT_SUPPORT"

    execution_pass=(
        len(exact)==10 and
        primary_support_pass and
        required_row_support_pass and
        numerical_pass and
        denominator_pass
    )

    # ---------------------------- artifacts ----------------------------

    with open(
        os.path.join(a.out,"P1_A7C_VALIDATION_SUPPORT.csv"),
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
        os.path.join(a.out,"P1_A7C_RECONSTRUCTION_DIAGNOSTICS.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=[
            "branch","configuration","file_name",
            "source_rows","common_envelope_rows","invalid_rows",
            "prop_thrust_reference_nrmse"
        ]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(diagnostics)

    branch_json=[]
    branch_csv=[]

    for name,z in fits.items():
        row={
            "branch":name,
            "pairs":z["pairs"],
            "jackknife_refits":z["jackknife_refits"],

            "CL_rank":z["CL_rank"],
            "CL_required_rank":z["CL_required_rank"],
            "CL_condition_number":z["CL_condition_number"],

            "CD_rank":z["CD_rank"],
            "CD_required_rank":z["CD_required_rank"],
            "CD_condition_number":z["CD_condition_number"],

            "largest_pair_fraction":z["largest_pair_fraction"],
            "denominator_gate":z["denominator_gate"],

            "CL_clean_ref":z["CL_clean_ref"],
            "CL_iced_ref":z["CL_iced_ref"],
            "delta_CL_ref":z["delta_CL_ref"],
            "delta_CL_jackknife_se":z["delta_CL_ref_jackknife_se"],
            "delta_CL_loo_min":z["delta_CL_ref_loo_min"],
            "delta_CL_loo_max":z["delta_CL_ref_loo_max"],

            "CD_clean_ref":z["CD_clean_ref"],
            "CD_iced_ref":z["CD_iced_ref"],
            "delta_CD_ref":z["delta_CD_ref"],
            "delta_CD_jackknife_se":z["delta_CD_ref_jackknife_se"],

            "E_clean_ref":z["E_clean_ref"],
            "E_iced_ref":z["E_iced_ref"],
            "delta_E_ref":z["delta_E_ref"],
            "delta_E_jackknife_se":z["delta_E_ref_jackknife_se"],

            "relative_efficiency_degradation":z["relative_efficiency_degradation"],
            "relative_efficiency_degradation_jackknife_se":
                z["relative_efficiency_degradation_jackknife_se"],
            "relative_efficiency_degradation_loo_min":
                z["relative_efficiency_degradation_loo_min"],
            "relative_efficiency_degradation_loo_max":
                z["relative_efficiency_degradation_loo_max"],

            "HE1_one_sided_p":z["HE1_one_sided_p"]
        }

        branch_json.append(row)
        branch_csv.append(row)

    with open(
        os.path.join(a.out,"P1_A7C_EFFICIENCY_VALIDATION_BRANCH_RESULTS.json"),
        "w",encoding="utf-8"
    ) as f:
        json.dump(branch_json,f,indent=2)

    with open(
        os.path.join(a.out,"P1_A7C_EFFICIENCY_VALIDATION_BRANCH_RESULTS.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=list(branch_csv[0].keys()) if branch_csv else []
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(branch_csv)

    audit={
        "scope":{
            "new_A7_validation_efficiency_outcomes_opened":True,
            "exact_longitudinal_validation_pairs":len(exact),
            "unmatched_longitudinal_validation_files":len(unmatched),
            "represented_validation_pairs":len(represented)
        },

        "support":{
            "primary_support_pass":primary_support_pass,
            "required_sensitivity_row_support_pass":required_row_support_pass,
            "required_sensitivity_row_support":sensitivity_row_support
        },

        "integrity":{
            "required_fit_numerical_pass":numerical_pass,
            "required_denominator_gate_pass":denominator_pass
        },

        "HE1":{
            "alternative":"delta_E_ref > 0",
            "relative_efficiency_degradation":
                None if primary is None else primary["relative_efficiency_degradation"],
            "jackknife_se":
                None if primary is None else primary["relative_efficiency_degradation_jackknife_se"],
            "one_sided_p":
                None if primary is None else primary["HE1_one_sided_p"]
        },

        "absolute_lift":{
            "CL_clean_ref":None if primary is None else primary["CL_clean_ref"],
            "CL_iced_ref":None if primary is None else primary["CL_iced_ref"],
            "delta_CL_ref":None if primary is None else primary["delta_CL_ref"],
            "delta_CL_jackknife_se":
                None if primary is None else primary["delta_CL_ref_jackknife_se"],
            "directional_confirmatory_claim":"NONE"
        },

        "required_directional_sensitivities":required_positive,
        "all_required_directional_sensitivities_positive":all_required_positive,

        "classification":classification,
        "W9_is_descriptive_only":True,
        "execution_pass":execution_pass
    }

    with open(
        os.path.join(a.out,"P1_A7C_AUDIT.json"),
        "w",encoding="utf-8"
    ) as f:
        json.dump(audit,f,indent=2)

    print("P1_A7C_LOCKED_REFERENCE_EFFICIENCY_VALIDATION_SUMMARY")
    print("exact_validation_pairs                        =",len(exact))
    print("unmatched_longitudinal_files                  =",len(unmatched))
    print("represented_validation_pairs                  =",len(represented))
    print("primary_support_pass                          =",primary_support_pass)
    print("required_sensitivity_row_support_pass         =",required_row_support_pass)
    print("required_fit_numerical_pass                   =",numerical_pass)
    print("required_denominator_gate_pass                =",denominator_pass)
    print("execution_pass                                =",execution_pass)
    print("classification                                =",classification)

    if primary is not None:
        print("")
        print("VALIDATION ABSOLUTE REFERENCE COEFFICIENTS")
        print("  C_L clean ref                               = {:+.9g}".format(primary["CL_clean_ref"]))
        print("  C_L iced ref                                = {:+.9g}".format(primary["CL_iced_ref"]))
        print("  Delta C_L ref                               = {:+.9g}".format(primary["delta_CL_ref"]))
        print("  Delta C_L jackknife SE                      = {:.9g}".format(primary["delta_CL_ref_jackknife_se"]))

        print("")
        print("  C_D clean ref                               = {:+.9g}".format(primary["CD_clean_ref"]))
        print("  C_D iced ref                                = {:+.9g}".format(primary["CD_iced_ref"]))
        print("  Delta C_D ref                               = {:+.9g}".format(primary["delta_CD_ref"]))

        print("")
        print("VALIDATION REFERENCE-STATE AERODYNAMIC EFFICIENCY")
        print("  E_clean_ref                                 = {:+.9g}".format(primary["E_clean_ref"]))
        print("  E_iced_ref                                  = {:+.9g}".format(primary["E_iced_ref"]))
        print("  Delta E_ref                                 = {:+.9g}".format(primary["delta_E_ref"]))
        print("  relative efficiency degradation             = {:+.3%}".format(primary["relative_efficiency_degradation"]))
        print("  jackknife SE(relative degradation)           = {:.9g}".format(primary["relative_efficiency_degradation_jackknife_se"]))
        print("  HE1 one-sided p                             = {:.9g}".format(primary["HE1_one_sided_p"]))

        print("")
        print("FROZEN VALIDATION EFFICIENCY SENSITIVITY BRANCHES")
        for name in (
            "PRIMARY_W5",
            "ETA1_W5",
            "EQUAL_PAIR_W5",
            "BETA_AUG_W5",
            "W7",
            "W9_STRESS"
        ):
            z=fits[name]
            gate="DESCRIPTIVE_ONLY" if name=="W9_STRESS" else (
                "POSITIVE_DEGRADATION"
                if z["relative_efficiency_degradation"]>0
                else "NONPOSITIVE_DEGRADATION"
            )

            print(
                "  {0:14s}: C_Lc={1:+.6g}, C_Li={2:+.6g}, "
                "C_Dc={3:+.6g}, C_Di={4:+.6g}, "
                "E_c={5:+.6g}, E_i={6:+.6g}, "
                "degradation={7:+.3%}, gate={8}".format(
                    name,
                    z["CL_clean_ref"],
                    z["CL_iced_ref"],
                    z["CD_clean_ref"],
                    z["CD_iced_ref"],
                    z["E_clean_ref"],
                    z["E_iced_ref"],
                    z["relative_efficiency_degradation"],
                    gate
                )
            )

if __name__=="__main__":
    main()
