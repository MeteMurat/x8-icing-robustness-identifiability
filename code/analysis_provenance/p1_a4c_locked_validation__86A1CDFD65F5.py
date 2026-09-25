import argparse,csv,json,math,os,re,statistics
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

def local_quad_derivative(t,y,i,half=2):
    lo=i-half
    hi=i+half+1
    if lo<0 or hi>len(t):
        return None
    tc=t[i]
    ts=[]
    ys=[]
    for k in range(lo,hi):
        if t[k] is None or y[k] is None:
            return None
        ts.append(t[k]-tc)
        ys.append(y[k])
    s0=len(ts)
    s1=sum(ts)
    s2=sum(x*x for x in ts)
    s3=sum(x*x*x for x in ts)
    s4=sum(x*x*x*x for x in ts)
    b0=sum(ys)
    b1=sum(x*v for x,v in zip(ts,ys))
    b2=sum(x*x*v for x,v in zip(ts,ys))
    a=solve3(
        [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]],
        [b0,b1,b2]
    )
    return a[1]

def cross(a,b):
    return (
        a[1]*b[2]-a[2]*b[1],
        a[2]*b[0]-a[0]*b[2],
        a[0]*b[1]-a[1]*b[0],
    )

def matvec(A,x):
    return tuple(sum(A[i][j]*x[j] for j in range(3)) for i in range(3))

def add(a,b):
    return tuple(a[i]+b[i] for i in range(3))

def sub(a,b):
    return tuple(a[i]-b[i] for i in range(3))

def scale(s,a):
    return tuple(s*x for x in a)

def poly(c,J):
    return c[0]+c[1]*J+c[2]*J*J+c[3]*J*J*J

def load_json(path):
    with open(path,"r",encoding="utf-8-sig") as f:
        return json.load(f)

def load_env(path):
    env={"lon":{},"lat":{}}
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            env[r["axis"]][r["variable"]]=(
                float(r["common_low"]),
                float(r["common_high"])
            )
    return env

def load_claims(path):
    out={}
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["claim_id"]]=r
    if set(out)!={"H1","H2","H3","H4"}:
        raise RuntimeError("Frozen claims must be exactly H1-H4.")
    return out

def validation_files(folder):
    out={}
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".csv"):
            continue
        m=re.search(r"_ID_(\d+)\.csv$",name,re.I)
        if m and int(m.group(1))>=4:
            out[name]=os.path.join(folder,name)
    return out

def axis_of(name):
    return "lon" if name.lower().startswith("lon_") else "lat"

def inside(x,env):
    return all(env[k][0] <= x[k] <= env[k][1] for k in env)

def reconstruct_file(path,cfg,pms,env_axis):
    m=float(pms["mass"])
    g=float(pms["gravity"])
    rho=float(pms["rho"])
    D=float(pms["D"])
    S=float(pms["S_wing"])
    b=float(pms["b"])
    cbar=float(pms["c"])
    Jp=float(pms["J_prop"])
    et=float(pms["eta_prop_T"])
    eq=float(pms["eta_prop_Q"])

    ct=[
        float(pms["C_T_0"]),
        float(pms["C_T_1"]),
        float(pms["C_T_2"]),
        float(pms["C_T_3"])
    ]
    cq=[
        float(pms["C_Q_0"]),
        float(pms["C_Q_1"]),
        float(pms["C_Q_2"]),
        float(pms["C_Q_3"])
    ]

    Jx=float(pms["Jx"])
    Jy=float(pms["Jy"])
    Jz=float(pms["Jz"])
    Jxz=float(pms["Jxz"])
    I=((Jx,0.0,-Jxz),(0.0,Jy,0.0),(-Jxz,0.0,Jz))

    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))

    def col(name):
        return [F(r.get(name)) for r in rows]

    t=col("TIME")
    u=col("u_CG")
    v=col("v_CG")
    w=col("w_CG")
    pr=col("p")
    qr=col("q")
    rr=col("r")
    pd=col("p_dot")
    qd=col("q_dot")
    rd=col("r_dot")
    Va=col("Va_EKF_CG")
    alpha=col("alpha")
    beta=col("beta")
    phi=col("phi")
    theta=col("theta")
    Om=col("RPS")
    elevator=col("elevator")
    aileron=col("aileron")

    axis=axis_of(os.path.basename(path))
    kept=[]
    invalid=0
    prop_ref_sq=0.0
    prop_ref_den=0.0
    prop_ref_n=0

    for i in range(2,len(rows)-2):
        needed=[
            t[i],u[i],v[i],w[i],
            pr[i],qr[i],rr[i],
            pd[i],qd[i],rd[i],
            Va[i],alpha[i],beta[i],
            phi[i],theta[i],Om[i],
            elevator[i],aileron[i]
        ]
        if any(x is None for x in needed):
            invalid+=1
            continue

        try:
            du=local_quad_derivative(t,u,i,2)
            dv=local_quad_derivative(t,v,i,2)
            dw=local_quad_derivative(t,w,i,2)
        except Exception:
            invalid+=1
            continue

        if None in (du,dv,dw):
            invalid+=1
            continue

        omega=(pr[i],qr[i],rr[i])
        vel=(u[i],v[i],w[i])
        vdot=(du,dv,dw)
        ain=add(vdot,cross(omega,vel))

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

        T=et*CT*rho*n*n*(D**4)
        Q=eq*CQ*rho*n*n*(D**5)

        Fprop=(T,0.0,0.0)
        Mprop=(-Q,-Jp*Om[i]*rr[i],+Jp*Om[i]*qr[i])

        Fa=sub(sub(scale(m,ain),Fprop),scale(m,gb))

        omega_dot=(pd[i],qd[i],rd[i])
        Iwd=matvec(I,omega_dot)
        Iw=matvec(I,omega)
        Ma=sub(add(Iwd,cross(omega,Iw)),Mprop)

        qbar=0.5*rho*Va[i]*Va[i]
        if not math.isfinite(qbar) or qbar<=1e-8:
            invalid+=1
            continue

        ca=math.cos(alpha[i])
        sa=math.sin(alpha[i])
        cb=math.cos(beta[i])
        sb=math.sin(beta[i])

        Lift=sa*Fa[0]-ca*Fa[2]
        CL=Lift/(qbar*S)

        Cm=Ma[1]/(qbar*S*cbar)
        Cl=Ma[0]/(qbar*S*b)
        Cn=Ma[2]/(qbar*S*b)

        # eta_prop = 1 sensitivity branch, with all other semantics frozen.
        T1=(CT*rho*n*n*(D**4))
        Q1=(CQ*rho*n*n*(D**5))

        dX=-(T1-T)
        CL_eta1=CL + sa*dX/(qbar*S)

        dL=(Q1-Q)
        Cl_eta1=Cl + dL/(qbar*S*b)

        q_hat=qr[i]*cbar/(2.0*Va[i])
        p_hat=pr[i]*b/(2.0*Va[i])
        r_hat=rr[i]*b/(2.0*Va[i])

        x={
            "Va":Va[i],
            "alpha":alpha[i],
            "beta":beta[i],
            "q_hat":q_hat,
            "p_hat":p_hat,
            "r_hat":r_hat,
            "delta_e":elevator[i],
            "delta_a":aileron[i]
        }

        envx={k:x[k] for k in env_axis}
        if not inside(envx,env_axis):
            continue

        srcT=F(rows[i].get("prop_thrust_bodyX"))
        if srcT is not None:
            prop_ref_sq+=(T-srcT)**2
            prop_ref_den+=srcT*srcT
            prop_ref_n+=1

        kept.append({
            "configuration":cfg,
            "file_name":os.path.basename(path),
            "sample_index":i,
            "axis":axis,
            **x,
            "C_L_lift":CL,
            "C_L_lift_eta1":CL_eta1,
            "C_m_pitch":Cm,
            "C_m_pitch_eta1":Cm,
            "C_l_roll":Cl,
            "C_l_roll_eta1":Cl_eta1,
            "C_n_yaw":Cn,
            "C_n_yaw_eta1":Cn
        })

    prop_nrmse=None
    if prop_ref_n and prop_ref_den>EPS:
        prop_nrmse=math.sqrt(prop_ref_sq/prop_ref_n)/math.sqrt(prop_ref_den/prop_ref_n)

    return {
        "source_rows":len(rows),
        "envelope_rows":len(kept),
        "invalid_rows":invalid,
        "prop_ref_nrmse":prop_nrmse,
        "rows":kept
    }

def demean(rows,predictors,outcome):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["file_name"],r["configuration"])].append(r)

    out=[]
    sizes={}
    for g,rr in groups.items():
        sizes[g]=len(rr)
        mx={k:sum(z[k] for z in rr)/len(rr) for k in predictors}
        my=sum(z[outcome] for z in rr)/len(rr)
        for z in rr:
            out.append({
                "configuration":z["configuration"],
                "file_name":z["file_name"],
                "sample_index":z["sample_index"],
                "_group":g,
                "_x":[z[k]-mx[k] for k in predictors],
                "_y":z[outcome]-my
            })
    return out,sizes

def fit_block(rows,predictors,outcome,weight_mode="row"):
    dm,sizes=demean(rows,predictors,outcome)
    p=len(predictors)

    X=[]
    Y=[]
    W=[]
    clusters=[]

    for r in dm:
        x=r["_x"]
        if r["configuration"]=="clean":
            vec=x+[0.0]*p
        else:
            vec=[0.0]*p+x
        X.append(vec)
        Y.append(r["_y"])
        clusters.append(r["file_name"])

        if weight_mode=="equal_flight":
            W.append(1.0/sizes[r["_group"]])
        else:
            W.append(1.0)

    X=np.asarray(X,float)
    Y=np.asarray(Y,float)
    W=np.asarray(W,float)

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

    unique_clusters=sorted(set(clusters))
    G=len(unique_clusters)
    carray=np.asarray(clusters,dtype=object)

    loo=[]
    failures=[]

    for g in unique_clusters:
        keep=carray!=g
        Xg=X[keep,:]
        Yg=Y[keep]
        Wg=W[keep]
        swg=np.sqrt(Wg)
        Xgw=Xg*swg[:,None]
        Ygw=Yg*swg
        sc=np.sqrt(np.mean(Xgw*Xgw,axis=0))
        if np.any(sc<=1e-14):
            failures.append(g)
            continue
        try:
            bg,_,rg,svg=np.linalg.lstsq(Xgw/sc,Ygw,rcond=None)
            if int(rg)!=2*p:
                failures.append(g)
                continue
            loo.append((g,bg/sc))
        except Exception:
            failures.append(g)

    B=np.vstack([x[1] for x in loo]) if loo else np.empty((0,2*p))

    def jk_stats(j):
        point=float(beta[j])
        if len(B)<2:
            return {"point":point,"se":None,"loo_min":None,"loo_max":None}
        vals=B[:,j]
        bar=float(np.mean(vals))
        se=math.sqrt((len(vals)-1.0)/len(vals)*float(np.sum((vals-bar)**2)))
        return {
            "point":point,
            "se":se,
            "loo_min":float(np.min(vals)),
            "loo_max":float(np.max(vals))
        }

    deriv={}
    for j,pred in enumerate(predictors):
        clean=jk_stats(j)
        iced=jk_stats(p+j)

        delta=float(beta[p+j]-beta[j])
        if len(B)>=2:
            dvals=B[:,p+j]-B[:,j]
            dbar=float(np.mean(dvals))
            dse=math.sqrt((len(dvals)-1.0)/len(dvals)*float(np.sum((dvals-dbar)**2)))
            dlo=float(np.min(dvals))
            dhi=float(np.max(dvals))
        else:
            dse=None
            dlo=None
            dhi=None

        deriv[pred]={
            "clean":clean["point"],
            "iced":iced["point"],
            "delta":delta,
            "se_clean":clean["se"],
            "se_iced":iced["se"],
            "se_delta":dse,
            "clean_loo_min":clean["loo_min"],
            "clean_loo_max":clean["loo_max"],
            "iced_loo_min":iced["loo_min"],
            "iced_loo_max":iced["loo_max"],
            "delta_loo_min":dlo,
            "delta_loo_max":dhi
        }

    cluster_sizes=defaultdict(int)
    for g in clusters:
        cluster_sizes[g]+=1

    largest_cluster_fraction=max(cluster_sizes.values())/len(clusters) if clusters else None

    return {
        "beta":beta,
        "rank":int(rank),
        "required_rank":2*p,
        "condition_number":cond,
        "clusters":G,
        "jackknife_refits":len(loo),
        "jackknife_failures":failures,
        "largest_cluster_fraction":largest_cluster_fraction,
        "derivatives":deriv
    }

def one_sided_p_pos(est,se,df):
    if se is None or se<=EPS:
        return 0.0 if est>0 else 1.0
    return float(student_t.sf(est/se,df))

def one_sided_p_neg(est,se,df):
    if se is None or se<=EPS:
        return 0.0 if est<0 else 1.0
    return float(student_t.cdf(est/se,df))

def holm_adjust(pvals):
    m=len(pvals)
    order=sorted(range(m),key=lambda i:pvals[i])
    adj=[1.0]*m
    running=0.0
    for rank0,idx in enumerate(order):
        val=(m-rank0)*pvals[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj

def bh_adjust(pvals):
    m=len(pvals)
    order=sorted(range(m),key=lambda i:pvals[i])
    adj=[1.0]*m
    running=1.0
    for rank0 in range(m-1,-1,-1):
        idx=order[rank0]
        rank=rank0+1
        val=pvals[idx]*m/rank
        running=min(running,val)
        adj[idx]=min(1.0,running)
    return adj

def sign_pattern_h2(d):
    return d["clean"]<0 and d["iced"]>0 and d["delta"]>0

def direction_for_claim(claim_id,d):
    if claim_id in ("H1","H3"):
        return d["delta"]>0
    if claim_id=="H4":
        return d["delta"]<0
    if claim_id=="H2":
        return sign_pattern_h2(d)
    raise KeyError(claim_id)

def claim_p(claim_id,d,df):
    if claim_id in ("H1","H3"):
        return one_sided_p_pos(d["delta"],d["se_delta"],df)
    if claim_id=="H4":
        return one_sided_p_neg(d["delta"],d["se_delta"],df)
    if claim_id=="H2":
        p_clean=one_sided_p_neg(d["clean"],d["se_clean"],df)
        p_iced=one_sided_p_pos(d["iced"],d["se_iced"],df)
        p_delta=one_sided_p_pos(d["delta"],d["se_delta"],df)
        # Intersection-union test for the predeclared three-part sign pattern.
        return max(p_clean,p_iced,p_delta)
    raise KeyError(claim_id)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clean-dir",required=True)
    ap.add_argument("--iced-dir",required=True)
    ap.add_argument("--clean-json",required=True)
    ap.add_argument("--iced-json",required=True)
    ap.add_argument("--envelope",required=True)
    ap.add_argument("--claims",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    env=load_env(a.envelope)
    claims=load_claims(a.claims)

    params={
        "clean":load_json(a.clean_json),
        "iced":load_json(a.iced_json)
    }

    files={
        "clean":validation_files(a.clean_dir),
        "iced":validation_files(a.iced_dir)
    }

    exact=sorted(set(files["clean"]).intersection(files["iced"]))
    unmatched=sorted(set(files["clean"]).symmetric_difference(files["iced"]))

    if len(exact)!=24:
        raise RuntimeError("Expected 24 exact validation pairs, observed %d"%len(exact))
    if len(unmatched)!=4:
        raise RuntimeError("Expected 4 unmatched validation files, observed %d"%len(unmatched))

    all_rows=[]
    pair_support=[]
    parse_errors=[]
    prop_ref=[]

    # First and only reconstruction pass through validation outcomes.
    for fn in exact:
        axis=axis_of(fn)
        eaxis=env[axis]

        res={}
        for cfg in ("clean","iced"):
            try:
                res[cfg]=reconstruct_file(
                    files[cfg][fn],
                    cfg,
                    params[cfg],
                    eaxis
                )
                prop_ref.append({
                    "configuration":cfg,
                    "file_name":fn,
                    "prop_thrust_reference_nrmse":res[cfg]["prop_ref_nrmse"]
                })
            except Exception as e:
                parse_errors.append({
                    "configuration":cfg,
                    "file_name":fn,
                    "error":repr(e)
                })
                res[cfg]={
                    "source_rows":0,
                    "envelope_rows":0,
                    "invalid_rows":0,
                    "prop_ref_nrmse":None,
                    "rows":[]
                }

        represented=(
            res["clean"]["envelope_rows"]>=25 and
            res["iced"]["envelope_rows"]>=25
        )

        pair_support.append({
            "file_name":fn,
            "axis":axis,
            "clean_source_rows":res["clean"]["source_rows"],
            "iced_source_rows":res["iced"]["source_rows"],
            "clean_common_envelope_rows":res["clean"]["envelope_rows"],
            "iced_common_envelope_rows":res["iced"]["envelope_rows"],
            "represented":represented
        })

        if represented:
            all_rows.extend(res["clean"]["rows"])
            all_rows.extend(res["iced"]["rows"])

    represented={
        "lon":[r["file_name"] for r in pair_support if r["axis"]=="lon" and r["represented"]],
        "lat":[r["file_name"] for r in pair_support if r["axis"]=="lat" and r["represented"]]
    }

    support_pre={
        axis:{
            "exact_pairs":sum(1 for r in pair_support if r["axis"]==axis),
            "represented_pairs":len(represented[axis]),
            "cluster_minimum_pass":len(represented[axis])>=6
        }
        for axis in ("lon","lat")
    }

    # Fit only frozen models required by H1-H4.
    model_defs={
        "H1":{
            "axis":"lon","outcome":"C_m_pitch",
            "predictors":["alpha","q_hat","delta_e"],
            "target":"alpha",
            "nuisance":["Va"],
            "eta1":False
        },
        "H2":{
            "axis":"lon","outcome":"C_L_lift",
            "predictors":["alpha","q_hat","delta_e"],
            "target":"delta_e",
            "nuisance":["Va"],
            "eta1":True
        },
        "H3":{
            "axis":"lat","outcome":"C_l_roll",
            "predictors":["beta","p_hat","r_hat","delta_a"],
            "target":"delta_a",
            "nuisance":["Va","alpha"],
            "eta1":True
        },
        "H4":{
            "axis":"lat","outcome":"C_n_yaw",
            "predictors":["beta","p_hat","r_hat","delta_a"],
            "target":"p_hat",
            "nuisance":["Va","alpha"],
            "eta1":False
        }
    }

    fits={}
    axis_design_pass={"lon":True,"lat":True}
    axis_design_details={"lon":[],"lat":[]}

    for cid,d in model_defs.items():
        axis=d["axis"]
        rr=[
            r for r in all_rows
            if r["axis"]==axis and r["file_name"] in represented[axis]
        ]

        if not support_pre[axis]["cluster_minimum_pass"]:
            fits[cid]=None
            axis_design_pass[axis]=False
            continue

        primary=fit_block(rr,d["predictors"],d["outcome"],"row")
        equalw=fit_block(rr,d["predictors"],d["outcome"],"equal_flight")
        augmented=fit_block(rr,d["predictors"]+d["nuisance"],d["outcome"],"row")

        eta1=None
        if d["eta1"]:
            eta1=fit_block(
                rr,
                d["predictors"],
                d["outcome"]+"_eta1",
                "row"
            )

        numerical=(
            primary["rank"]==primary["required_rank"] and
            primary["condition_number"]<=30.0 and
            primary["clusters"]>=6 and
            primary["largest_cluster_fraction"]<=0.25 and
            primary["jackknife_refits"]==primary["clusters"] and
            equalw["rank"]==equalw["required_rank"] and
            equalw["condition_number"]<=30.0 and
            augmented["rank"]==augmented["required_rank"] and
            augmented["condition_number"]<=30.0 and
            (
                eta1 is None or
                (
                    eta1["rank"]==eta1["required_rank"] and
                    eta1["condition_number"]<=30.0
                )
            )
        )

        axis_design_pass[axis]=axis_design_pass[axis] and numerical

        axis_design_details[axis].append({
            "claim_id":cid,
            "primary_rank":primary["rank"],
            "primary_required_rank":primary["required_rank"],
            "primary_condition_number":primary["condition_number"],
            "clusters":primary["clusters"],
            "largest_cluster_fraction":primary["largest_cluster_fraction"],
            "jackknife_refits":primary["jackknife_refits"],
            "equal_weight_condition_number":equalw["condition_number"],
            "augmented_condition_number":augmented["condition_number"],
            "eta1_condition_number":None if eta1 is None else eta1["condition_number"],
            "numerical_pass":numerical
        })

        fits[cid]={
            "primary":primary,
            "equal_weight":equalw,
            "augmented":augmented,
            "eta1":eta1
        }

    axis_support={
        axis:(
            support_pre[axis]["cluster_minimum_pass"] and
            axis_design_pass[axis]
        )
        for axis in ("lon","lat")
    }

    raw_p=[]
    provisional=[]

    for cid in ("H1","H2","H3","H4"):
        d=model_defs[cid]
        axis=d["axis"]

        if not axis_support[axis] or fits[cid] is None:
            provisional.append({
                "claim_id":cid,
                "axis":axis,
                "derivative":claims[cid]["derivative"],
                "support_pass":False,
                "primary_direction_replicated":False,
                "sensitivity_direction_stable":False,
                "raw_one_sided_p":1.0,
                "classification_pre_multiplicity":"INSUFFICIENT_SUPPORT",
                "primary":None,
                "equal_weight":None,
                "augmented":None,
                "eta1":None
            })
            raw_p.append(1.0)
            continue

        target=d["target"]
        primary=fits[cid]["primary"]["derivatives"][target]
        ew=fits[cid]["equal_weight"]["derivatives"][target]
        aug=fits[cid]["augmented"]["derivatives"][target]
        eta=(
            None if fits[cid]["eta1"] is None
            else fits[cid]["eta1"]["derivatives"][target]
        )

        df=fits[cid]["primary"]["clusters"]-1
        pval=claim_p(cid,primary,df)
        primary_direction=direction_for_claim(cid,primary)

        sens_dirs=[
            direction_for_claim(cid,ew),
            direction_for_claim(cid,aug)
        ]
        if eta is not None:
            sens_dirs.append(direction_for_claim(cid,eta))

        sens_stable=all(sens_dirs)

        provisional.append({
            "claim_id":cid,
            "axis":axis,
            "derivative":claims[cid]["derivative"],
            "support_pass":True,
            "primary_direction_replicated":primary_direction,
            "sensitivity_direction_stable":sens_stable,
            "raw_one_sided_p":pval,
            "classification_pre_multiplicity":"PENDING",
            "primary":primary,
            "equal_weight":ew,
            "augmented":aug,
            "eta1":eta
        })
        raw_p.append(pval)

    holm=holm_adjust(raw_p)
    bh=bh_adjust(raw_p)

    results=[]

    for i,z in enumerate(provisional):
        if not z["support_pass"]:
            cls="INSUFFICIENT_SUPPORT"
        elif not z["primary_direction_replicated"]:
            cls="NOT_REPLICATED"
        elif not z["sensitivity_direction_stable"]:
            cls="MODEL_SENSITIVE_VALIDATION"
        elif holm[i]<=0.05:
            cls="CONFIRMED_STRONG"
        else:
            cls="DIRECTIONALLY_REPLICATED"

        p=z["primary"]
        ew=z["equal_weight"]
        aug=z["augmented"]
        eta=z["eta1"]

        result={
            "claim_id":z["claim_id"],
            "axis":z["axis"],
            "derivative":z["derivative"],
            "support_pass":z["support_pass"],
            "represented_validation_clusters":len(represented[z["axis"]]),
            "primary_clean":None if p is None else p["clean"],
            "primary_iced":None if p is None else p["iced"],
            "primary_delta":None if p is None else p["delta"],
            "primary_se_clean":None if p is None else p["se_clean"],
            "primary_se_iced":None if p is None else p["se_iced"],
            "primary_se_delta":None if p is None else p["se_delta"],
            "primary_direction_replicated":z["primary_direction_replicated"],
            "equal_weight_clean":None if ew is None else ew["clean"],
            "equal_weight_iced":None if ew is None else ew["iced"],
            "equal_weight_delta":None if ew is None else ew["delta"],
            "augmented_clean":None if aug is None else aug["clean"],
            "augmented_iced":None if aug is None else aug["iced"],
            "augmented_delta":None if aug is None else aug["delta"],
            "eta1_clean":None if eta is None else eta["clean"],
            "eta1_iced":None if eta is None else eta["iced"],
            "eta1_delta":None if eta is None else eta["delta"],
            "sensitivity_direction_stable":z["sensitivity_direction_stable"],
            "raw_one_sided_p":z["raw_one_sided_p"],
            "holm_adjusted_p":holm[i],
            "bh_adjusted_p":bh[i],
            "classification":cls,
            "identification_clean":F(claims[z["claim_id"]].get("identification_clean")),
            "identification_iced":F(claims[z["claim_id"]].get("identification_iced")),
            "identification_delta":F(claims[z["claim_id"]].get("identification_delta"))
        }

        # Descriptive validation consequence magnitude only; not an extra hypothesis.
        if p is not None:
            if z["claim_id"]=="H1" and abs(p["clean"])>EPS:
                result["validation_consequence_fraction"]=1.0-abs(p["iced"])/abs(p["clean"])
            elif z["claim_id"] in ("H3","H4") and abs(p["clean"])>EPS:
                result["validation_consequence_fraction"]=abs(p["iced"])/abs(p["clean"])-1.0
            elif z["claim_id"]=="H2":
                result["validation_consequence_fraction"]=1.0 if sign_pattern_h2(p) else 0.0
            else:
                result["validation_consequence_fraction"]=None
        else:
            result["validation_consequence_fraction"]=None

        results.append(result)

    # Execution integrity is deliberately independent of whether hypotheses replicate.
    all_supported_models_numerical=True
    for axis in ("lon","lat"):
        if support_pre[axis]["cluster_minimum_pass"] and not axis_design_pass[axis]:
            all_supported_models_numerical=False

    max_prop_nrmse=max(
        [
            r["prop_thrust_reference_nrmse"]
            for r in prop_ref
            if r["prop_thrust_reference_nrmse"] is not None
        ] or [0.0]
    )

    execution_pass=(
        len(parse_errors)==0 and
        len(exact)==24 and
        len(unmatched)==4 and
        all_supported_models_numerical
    )

    # Write support inventory.
    with open(
        os.path.join(a.out,"P1_A4C_VALIDATION_SUPPORT.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=[
            "file_name","axis",
            "clean_source_rows","iced_source_rows",
            "clean_common_envelope_rows","iced_common_envelope_rows",
            "represented"
        ]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(pair_support)

    with open(
        os.path.join(a.out,"P1_A4C_CLAIM_RESULTS.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=list(results[0].keys())
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    model_rows=[]
    for axis in ("lon","lat"):
        model_rows.extend(axis_design_details[axis])

    with open(
        os.path.join(a.out,"P1_A4C_MODEL_INTEGRITY.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=[
            "claim_id",
            "primary_rank","primary_required_rank","primary_condition_number",
            "clusters","largest_cluster_fraction","jackknife_refits",
            "equal_weight_condition_number","augmented_condition_number",
            "eta1_condition_number","numerical_pass"
        ]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(model_rows)

    with open(
        os.path.join(a.out,"P1_A4C_PROPULSION_REFERENCE_DIAGNOSTIC.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=["configuration","file_name","prop_thrust_reference_nrmse"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(prop_ref)

    with open(
        os.path.join(a.out,"P1_A4C_PARSE_ERRORS.csv"),
        "w",newline="",encoding="utf-8-sig"
    ) as f:
        fields=["configuration","file_name","error"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(parse_errors)

    audit={
        "scope":{
            "validation_outcomes_opened":True,
            "exact_validation_pairs":len(exact),
            "unmatched_validation_files":len(unmatched),
            "validation_outcome_rows_reconstructed":sum(
                r["clean_common_envelope_rows"]+r["iced_common_envelope_rows"]
                for r in pair_support
            ),
            "validation_rows_used_in_confirmatory_models":len(all_rows)
        },
        "support":{
            "longitudinal":support_pre["lon"],
            "lateral":support_pre["lat"],
            "axis_design_pass":axis_design_pass,
            "axis_support_pass":axis_support
        },
        "integrity":{
            "parse_errors":len(parse_errors),
            "supported_model_numerical_integrity":all_supported_models_numerical,
            "propulsion_reference_max_nrmse":max_prop_nrmse
        },
        "multiplicity":{
            "family_size":4,
            "method":"Holm FWER alpha=0.05",
            "raw_one_sided_p":raw_p,
            "holm_adjusted_p":holm,
            "bh_adjusted_p_secondary":bh
        },
        "claim_classifications":{
            r["claim_id"]:r["classification"] for r in results
        },
        "execution_pass":execution_pass
    }

    with open(
        os.path.join(a.out,"P1_A4C_VALIDATION_AUDIT.json"),
        "w",encoding="utf-8"
    ) as f:
        json.dump(audit,f,indent=2)

    print("P1_A4C_LOCKED_VALIDATION_SUMMARY")
    print("exact_validation_pairs             =",len(exact))
    print("unmatched_validation_files         =",len(unmatched))
    print("represented_lon_pairs              =",len(represented["lon"]),"/",support_pre["lon"]["exact_pairs"])
    print("represented_lat_pairs              =",len(represented["lat"]),"/",support_pre["lat"]["exact_pairs"])
    print("axis_support_lon                   =",axis_support["lon"])
    print("axis_support_lat                   =",axis_support["lat"])
    print("parse_errors                       =",len(parse_errors))
    print("propulsion_reference_max_nrmse     =",max_prop_nrmse)
    print("execution_integrity_pass           =",execution_pass)
    print("")
    print("CONFIRMATORY CLAIM RESULTS")
    for r in results:
        print(
            "  {0}: {1:24s} G={2:2d} "
            "clean={3} iced={4} delta={5} "
            "p1={6:.6g} Holm={7:.6g} sens={8}".format(
                r["claim_id"],
                r["classification"],
                r["represented_validation_clusters"],
                "NA" if r["primary_clean"] is None else "{:+.6g}".format(r["primary_clean"]),
                "NA" if r["primary_iced"] is None else "{:+.6g}".format(r["primary_iced"]),
                "NA" if r["primary_delta"] is None else "{:+.6g}".format(r["primary_delta"]),
                r["raw_one_sided_p"],
                r["holm_adjusted_p"],
                r["sensitivity_direction_stable"]
            )
        )

if __name__=="__main__":
    main()
