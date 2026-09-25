import argparse,csv,json,math,os,re
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

EPS=1e-14

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
        raise RuntimeError("Frozen longitudinal B0 envelope incomplete.")
    return env

def id_files(folder):
    out={}
    for name in sorted(os.listdir(folder)):
        if not name.lower().startswith("lon_") or not name.lower().endswith(".csv"):
            continue
        m=re.search(r"_ID_(\d+)\.csv$",name,re.I)
        if m and int(m.group(1))<=3:
            out[name]=os.path.join(folder,name)
    return out

def inside_state(Va,alpha,qhat,de,env):
    return (
        env["Va"][0] <= Va <= env["Va"][1] and
        env["alpha"][0] <= alpha <= env["alpha"][1] and
        env["q_hat"][0] <= qhat <= env["q_hat"][1] and
        env["delta_e"][0] <= de <= env["delta_e"][1]
    )

def read_file(path,cfg,cbar,env):
    with open(path,"r",newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))

    out=[]
    eligible_state=0
    finite_batt=0

    for r in rows:
        Va=F(r.get("Va_EKF_CG"))
        alpha=F(r.get("alpha"))
        q=F(r.get("q"))
        de=F(r.get("elevator"))

        if None in (Va,alpha,q,de) or Va <= EPS:
            continue

        qhat=q*cbar/(2.0*Va)

        if not inside_state(Va,alpha,qhat,de,env):
            continue

        eligible_state += 1

        batt=F(r.get("battery_power"))
        if batt is None:
            continue

        finite_batt += 1

        out.append({
            "configuration":cfg,
            "Va":Va,
            "alpha":alpha,
            "q_hat":qhat,
            "delta_e":de,
            "beta":F(r.get("beta")),
            "Va_dot":F(r.get("Va_EKF_CG_dot")),
            "theta":F(r.get("theta")),
            "battery_power":batt,
            "throttle":F(r.get("throttle")),
            "RPS":F(r.get("RPS")),
            "voltage":F(r.get("voltage")),
            "current":F(r.get("current"))
        })

    finiteness=(finite_batt/eligible_state) if eligible_state else None

    return {
        "source_rows":len(rows),
        "eligible_state_rows":eligible_state,
        "finite_battery_rows":finite_batt,
        "battery_finiteness":finiteness,
        "rows":out
    }

def with_pair(rows,pair):
    out=[]
    for r in rows:
        z=dict(r)
        z["file_name"]=pair
        out.append(z)
    return out

def build_model_rows(raw,ref,extra_mode=None,extra_ref=None):
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

        if extra_mode=="beta":
            if r["beta"] is None:
                continue
            z["beta_c"]=r["beta"]

        elif extra_mode=="kinematic":
            if r["Va_dot"] is None or r["theta"] is None:
                continue
            z["Va_dot_c"]=r["Va_dot"]
            z["gamma_proxy_c"]=(r["theta"]-r["alpha"])

        elif extra_mode=="throttle":
            if r["throttle"] is None:
                continue
            z["throttle_c"]=r["throttle"]-extra_ref

        elif extra_mode=="rps":
            if r["RPS"] is None:
                continue
            z["RPS_c"]=r["RPS"]-extra_ref

        out.append(z)

    return out

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

def core_fit(raw,ref,weight_mode="row",extra_mode=None,extra_ref=None):
    rows=build_model_rows(raw,ref,extra_mode,extra_ref)

    base=["Va_c","alpha_c","alpha_c2","q_hat_c","delta_e_c"]
    cols=base+[
        "ice",
        "ice_Va_c","ice_alpha_c","ice_alpha_c2","ice_q_hat_c","ice_delta_e_c"
    ]

    if extra_mode=="beta":
        cols=["beta_c"]+cols
    elif extra_mode=="kinematic":
        cols=["Va_dot_c","gamma_proxy_c"]+cols
    elif extra_mode=="throttle":
        cols=["throttle_c"]+cols
    elif extra_mode=="rps":
        cols=["RPS_c"]+cols

    if len(rows)==0:
        raise RuntimeError("No rows available for model branch.")

    X,Y,W,clusters=pair_demean(rows,cols,"battery_power",weight_mode)

    if X.shape[0] <= X.shape[1]:
        raise RuntimeError("Insufficient rows for model rank.")

    sw=np.sqrt(W)
    Xw=X*sw[:,None]
    Yw=Y*sw

    scales=np.sqrt(np.mean(Xw*Xw,axis=0))
    if np.any(scales<=1e-14):
        raise RuntimeError("zero design scale")

    Xs=Xw/scales
    bs,_,rank,svals=np.linalg.lstsq(Xs,Yw,rcond=None)
    beta=bs/scales

    if len(svals)==0 or np.min(svals)<=EPS:
        cond=float("inf")
    else:
        cond=float(np.max(svals)/np.min(svals))

    gamma_idx=cols.index("ice")
    gamma0=float(beta[gamma_idx])

    # Reconstruct pair fixed effects on original scale.
    by_pair=defaultdict(list)
    for r in rows:
        pred=sum(beta[j]*r[cols[j]] for j in range(len(cols)))
        by_pair[r["file_name"]].append((r["battery_power"],pred))

    pair_intercepts={}
    for pair,items in by_pair.items():
        pair_intercepts[pair]=sum(y-pred for y,pred in items)/len(items)

    clean_ref=float(np.mean(list(pair_intercepts.values())))
    iced_ref=clean_ref+gamma0
    relative=(gamma0/clean_ref) if clean_ref>EPS else None

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
        "pairs":len(set(clusters.tolist())),
        "largest_pair_fraction":max_cluster,
        "gamma0":gamma0,
        "clean_ref":clean_ref,
        "iced_ref":iced_ref,
        "relative_penalty":relative,
        "row_count":len(rows)
    }

def jackknife_fit(raw,ref,weight_mode="row",extra_mode=None,extra_ref=None):
    full=core_fit(raw,ref,weight_mode,extra_mode,extra_ref)
    pairs=sorted(set(r["file_name"] for r in raw))
    loo=[]
    failures=[]

    for g in pairs:
        rr=[r for r in raw if r["file_name"]!=g]
        try:
            z=core_fit(rr,ref,weight_mode,extra_mode,extra_ref)
            loo.append({
                "left_out_pair":g,
                "gamma0":z["gamma0"],
                "clean_ref":z["clean_ref"],
                "iced_ref":z["iced_ref"],
                "relative_penalty":z["relative_penalty"]
            })
        except Exception as e:
            failures.append({"left_out_pair":g,"error":str(e)})

    def jk_se(vals):
        vals=[v for v in vals if v is not None and math.isfinite(v)]
        if len(vals)<2:
            return None
        a=np.asarray(vals,float)
        bar=float(np.mean(a))
        return math.sqrt((len(a)-1.0)/len(a)*float(np.sum((a-bar)**2)))

    full["jackknife_refits"]=len(loo)
    full["jackknife_failures"]=failures
    full["gamma0_jackknife_se"]=jk_se([x["gamma0"] for x in loo])
    full["relative_penalty_jackknife_se"]=jk_se([x["relative_penalty"] for x in loo])
    full["gamma0_loo_min"]=min([x["gamma0"] for x in loo]) if loo else None
    full["gamma0_loo_max"]=max([x["gamma0"] for x in loo]) if loo else None
    full["clean_ref_positive_all_loo"]=all(x["clean_ref"]>0 for x in loo) if loo else False
    full["iced_ref_positive_all_loo"]=all(x["iced_ref"]>0 for x in loo) if loo else False
    full["relative_denominator_positive_all_loo"]=all(x["clean_ref"]>0 for x in loo) if loo else False
    full["loo"]=loo
    return full

def one_sided_positive_p(est,se,df):
    if se is None or se<=EPS:
        return 0.0 if est>0 else 1.0
    return float(student_t.sf(est/se,df))

def voltage_current_diagnostic(rows):
    vals=[]
    for r in rows:
        if r["voltage"] is None or r["current"] is None:
            continue
        pvi=r["voltage"]*r["current"]
        pb=r["battery_power"]
        if not math.isfinite(pvi) or not math.isfinite(pb):
            continue
        vals.append((pb,pvi))

    if not vals:
        return {
            "n":0,
            "pearson_r":None,
            "same_sign_fraction":None,
            "median_pvi_over_battery_power":None,
            "median_pvi_minus_battery_power_W":None
        }

    pb=np.asarray([x[0] for x in vals],float)
    pv=np.asarray([x[1] for x in vals],float)

    if len(vals)>=2 and np.std(pb)>EPS and np.std(pv)>EPS:
        corr=float(np.corrcoef(pb,pv)[0,1])
    else:
        corr=None

    ratios=[b/a for a,b in vals if abs(a)>EPS]

    return {
        "n":len(vals),
        "pearson_r":corr,
        "same_sign_fraction":float(np.mean(np.sign(pb)==np.sign(pv))),
        "median_pvi_over_battery_power":float(np.median(ratios)) if ratios else None,
        "median_pvi_minus_battery_power_W":float(np.median(pv-pb))
    }

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

    pre=load_json(a.preanalysis)
    env=load_env(a.envelope)
    clean_p=load_json(a.clean_json)
    iced_p=load_json(a.iced_json)

    c_clean=float(clean_p["c"])
    c_iced=float(iced_p["c"])
    if abs(c_clean-c_iced)>1e-12:
        raise RuntimeError("clean/iced reference chord mismatch")
    cbar=c_clean

    ref={
        "Va":float(pre["frozen_reference_state"]["Va_ref_m_s"]),
        "alpha":float(pre["frozen_reference_state"]["alpha_ref_rad"]),
        "q_hat":float(pre["frozen_reference_state"]["q_hat_ref"]),
        "delta_e":float(pre["frozen_reference_state"]["delta_e_ref_rad"])
    }

    files={"clean":id_files(a.clean_dir),"iced":id_files(a.iced_dir)}
    exact=sorted(set(files["clean"]).intersection(files["iced"]))

    if len(exact)!=12:
        raise RuntimeError("Expected 12 exact ID1-3 longitudinal pairs; observed %d"%len(exact))

    pair_data={}
    support=[]
    all_primary_rows=[]

    total_eligible=0
    total_finite=0

    for fn in exact:
        pair_data[fn]={}
        row_support={"file_name":fn}
        represented=True

        for cfg in ("clean","iced"):
            res=read_file(files[cfg][fn],cfg,cbar,env)
            rows=with_pair(res["rows"],fn)
            pair_data[fn][cfg]=rows

            row_support[cfg+"_source_rows"]=res["source_rows"]
            row_support[cfg+"_state_envelope_rows"]=res["eligible_state_rows"]
            row_support[cfg+"_finite_battery_rows"]=res["finite_battery_rows"]
            row_support[cfg+"_battery_finiteness"]=res["battery_finiteness"]

            total_eligible += res["eligible_state_rows"]
            total_finite += res["finite_battery_rows"]

            if res["finite_battery_rows"] < 25:
                represented=False

        row_support["represented_primary"]=represented
        support.append(row_support)

        if represented:
            all_primary_rows.extend(pair_data[fn]["clean"])
            all_primary_rows.extend(pair_data[fn]["iced"])

    represented=[r["file_name"] for r in support if r["represented_primary"]]
    primary_support_pass=(len(represented)>=10)

    global_battery_finiteness=(total_finite/total_eligible) if total_eligible else None
    battery_finiteness_pass=(
        global_battery_finiteness is not None and
        global_battery_finiteness>=0.99
    )

    # Restrict all branches to the same primary represented exact-pair set.
    raw=[]
    for fn in represented:
        raw.extend(pair_data[fn]["clean"])
        raw.extend(pair_data[fn]["iced"])

    fits={}
    branch_errors={}

    # Predictor reference values for descriptive mediator branches are common pooled values.
    throttle_vals=[r["throttle"] for r in raw if r["throttle"] is not None]
    rps_vals=[r["RPS"] for r in raw if r["RPS"] is not None]
    throttle_ref=float(np.mean(throttle_vals)) if throttle_vals else None
    rps_ref=float(np.mean(rps_vals)) if rps_vals else None

    specs={
        "PRIMARY":{"weight_mode":"row","extra_mode":None,"extra_ref":None,"required":True},
        "EQUAL_PAIR":{"weight_mode":"equal_pair","extra_mode":None,"extra_ref":None,"required":True},
        "BETA_AUG":{"weight_mode":"row","extra_mode":"beta","extra_ref":None,"required":True},
        "KINEMATIC_ENERGY_STATE_AUG":{"weight_mode":"row","extra_mode":"kinematic","extra_ref":None,"required":True},
        "THROTTLE_AUG":{"weight_mode":"row","extra_mode":"throttle","extra_ref":throttle_ref,"required":False},
        "RPS_AUG":{"weight_mode":"row","extra_mode":"rps","extra_ref":rps_ref,"required":False}
    }

    if primary_support_pass:
        for name,s in specs.items():
            try:
                fits[name]=jackknife_fit(
                    raw,ref,
                    weight_mode=s["weight_mode"],
                    extra_mode=s["extra_mode"],
                    extra_ref=s["extra_ref"]
                )
            except Exception as e:
                branch_errors[name]=str(e)

    required_names=["PRIMARY","EQUAL_PAIR","BETA_AUG","KINEMATIC_ENERGY_STATE_AUG"]

    numerical_pass=primary_support_pass
    if primary_support_pass:
        for name in required_names:
            if name not in fits:
                numerical_pass=False
                continue
            z=fits[name]
            numerical_pass = numerical_pass and (
                z["rank"]==z["required_rank"] and
                z["condition_number"]<=30.0 and
                z["pairs"]>=10 and
                z["largest_pair_fraction"]<=0.20 and
                z["jackknife_refits"]==z["pairs"]
            )

    sign_semantic_pass=False
    relative_denominator_pass=False
    primary=None
    one_sided_p=None

    if "PRIMARY" in fits:
        primary=fits["PRIMARY"]
        sign_semantic_pass=(
            primary["clean_ref"]>0 and
            primary["iced_ref"]>0 and
            primary["clean_ref_positive_all_loo"] and
            primary["iced_ref_positive_all_loo"]
        )
        relative_denominator_pass=(
            primary["clean_ref"]>0 and
            primary["relative_denominator_positive_all_loo"]
        )
        one_sided_p=one_sided_positive_p(
            primary["gamma0"],
            primary["gamma0_jackknife_se"],
            primary["pairs"]-1
        )

    required_positive={}
    for name in required_names:
        required_positive[name]=(name in fits and fits[name]["gamma0"]>0)

    all_required_positive=all(required_positive.values()) if required_positive else False

    science_gate=(
        primary_support_pass and
        numerical_pass and
        battery_finiteness_pass and
        sign_semantic_pass
    )

    if not science_gate:
        classification="INSUFFICIENT_IDENTIFICATION_SUPPORT_OR_SEMANTIC_GATE"
    elif primary["gamma0"]<=0:
        classification="NO_REFERENCE_BATTERY_POWER_INCREASE"
    elif not all_required_positive:
        classification="BATTERY_POWER_SIGNAL_MODEL_SENSITIVE"
    else:
        classification="BATTERY_POWER_SIGNAL_ROBUST_IDENTIFICATION"

    execution_pass=(
        len(exact)==12 and
        primary_support_pass and
        numerical_pass
    )

    vi=voltage_current_diagnostic(raw)

    # Artifacts.
    with open(os.path.join(a.out,"P1_A8C_PAIR_SUPPORT.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=[
            "file_name",
            "clean_source_rows","clean_state_envelope_rows","clean_finite_battery_rows","clean_battery_finiteness",
            "iced_source_rows","iced_state_envelope_rows","iced_finite_battery_rows","iced_battery_finiteness",
            "represented_primary"
        ]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in support:
            w.writerow(r)

    branch_rows=[]
    coef_rows=[]
    for name,s in specs.items():
        if name not in fits:
            branch_rows.append({
                "branch":name,
                "required_claim_gate":s["required"],
                "fit_status":"FAILED",
                "error":branch_errors.get(name,""),
                "pairs":None,"row_count":None,"rank":None,"required_rank":None,
                "condition_number":None,"largest_pair_fraction":None,"jackknife_refits":None,
                "delta_P_batt_ref_W":None,"jackknife_se_W":None,"one_sided_p":None,
                "P_batt_clean_ref_W":None,"P_batt_iced_ref_W":None,
                "relative_penalty":None,"relative_penalty_jackknife_se":None,
                "gamma0_loo_min_W":None,"gamma0_loo_max_W":None
            })
            continue

        z=fits[name]
        p=one_sided_positive_p(z["gamma0"],z["gamma0_jackknife_se"],z["pairs"]-1)
        branch_rows.append({
            "branch":name,
            "required_claim_gate":s["required"],
            "fit_status":"OK",
            "error":"",
            "pairs":z["pairs"],
            "row_count":z["row_count"],
            "rank":z["rank"],
            "required_rank":z["required_rank"],
            "condition_number":z["condition_number"],
            "largest_pair_fraction":z["largest_pair_fraction"],
            "jackknife_refits":z["jackknife_refits"],
            "delta_P_batt_ref_W":z["gamma0"],
            "jackknife_se_W":z["gamma0_jackknife_se"],
            "one_sided_p":p,
            "P_batt_clean_ref_W":z["clean_ref"],
            "P_batt_iced_ref_W":z["iced_ref"],
            "relative_penalty":z["relative_penalty"],
            "relative_penalty_jackknife_se":z["relative_penalty_jackknife_se"],
            "gamma0_loo_min_W":z["gamma0_loo_min"],
            "gamma0_loo_max_W":z["gamma0_loo_max"]
        })

        for k,v in z["coefficients"].items():
            coef_rows.append({"branch":name,"coefficient":k,"value":v})

    with open(os.path.join(a.out,"P1_A8C_BATTERY_POWER_BRANCH_RESULTS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=[
            "branch","required_claim_gate","fit_status","error",
            "pairs","row_count","rank","required_rank","condition_number",
            "largest_pair_fraction","jackknife_refits",
            "delta_P_batt_ref_W","jackknife_se_W","one_sided_p",
            "P_batt_clean_ref_W","P_batt_iced_ref_W",
            "relative_penalty","relative_penalty_jackknife_se",
            "gamma0_loo_min_W","gamma0_loo_max_W"
        ]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in branch_rows:
            w.writerow(r)

    with open(os.path.join(a.out,"P1_A8C_BATTERY_POWER_COEFFICIENTS.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=["branch","coefficient","value"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in coef_rows:
            w.writerow(r)

    with open(os.path.join(a.out,"P1_A8C_VOLTAGE_CURRENT_DIAGNOSTIC.csv"),"w",newline="",encoding="utf-8-sig") as f:
        fields=list(vi.keys())
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerow(vi)

    audit={
        "scope":{
            "exact_longitudinal_identification_pairs":len(exact),
            "represented_primary_pairs":len(represented),
            "represented_pair_names":represented,
            "validation_battery_power_outcomes_opened":False
        },
        "support":{
            "minimum_pairs":10,
            "minimum_rows_each_configuration_per_pair":25,
            "primary_support_pass":primary_support_pass,
            "global_state_eligible_rows":total_eligible,
            "global_finite_battery_rows":total_finite,
            "global_battery_power_finiteness":global_battery_finiteness,
            "battery_power_finiteness_pass":battery_finiteness_pass
        },
        "integrity":{
            "required_fit_numerical_pass":numerical_pass,
            "sign_semantic_pass":sign_semantic_pass,
            "relative_denominator_pass":relative_denominator_pass,
            "execution_pass":execution_pass,
            "branch_errors":branch_errors
        },
        "hypothesis":{
            "id":"HE_B1",
            "alternative":"Delta P_batt,ref > 0 W",
            "primary_delta_P_batt_ref_W":None if primary is None else primary["gamma0"],
            "primary_jackknife_se_W":None if primary is None else primary["gamma0_jackknife_se"],
            "one_sided_p":one_sided_p,
            "required_branch_positive":required_positive,
            "all_required_positive":all_required_positive
        },
        "reference":{
            "P_batt_clean_ref_W":None if primary is None else primary["clean_ref"],
            "P_batt_iced_ref_W":None if primary is None else primary["iced_ref"],
            "relative_penalty":None if primary is None else primary["relative_penalty"],
            "relative_penalty_jackknife_se":None if primary is None else primary["relative_penalty_jackknife_se"],
            "throttle_common_pooled_reference":throttle_ref,
            "RPS_common_pooled_reference":rps_ref
        },
        "classification":classification,
        "voltage_current_diagnostic":vi,
        "execution_pass":execution_pass
    }

    with open(os.path.join(a.out,"P1_A8C_AUDIT.json"),"w",encoding="utf-8") as f:
        json.dump(audit,f,indent=2,sort_keys=True)

    print("P1_A8C_IDENTIFICATION_BATTERY_POWER_SUMMARY")
    print("exact_longitudinal_pairs            =",len(exact))
    print("represented_primary_pairs           =",len(represented))
    print("battery_power_finiteness            =",global_battery_finiteness)
    print("primary_support_pass                =",primary_support_pass)
    print("required_fit_numerical_pass         =",numerical_pass)
    print("sign_semantic_pass                  =",sign_semantic_pass)
    print("execution_pass                      =",execution_pass)
    print("classification                      =",classification)

    if primary is not None:
        print("P_batt_clean_ref_W                  =",primary["clean_ref"])
        print("P_batt_iced_ref_W                   =",primary["iced_ref"])
        print("Delta_P_batt_ref_W                  =",primary["gamma0"])
        print("jackknife_SE_W                      =",primary["gamma0_jackknife_se"])
        print("one_sided_p                         =",one_sided_p)
        print("relative_penalty                    =",primary["relative_penalty"])

    for name in specs:
        if name in fits:
            z=fits[name]
            print(
                "BRANCH {:30s} DeltaP={:.9g} cond={:.6g} rank={}/{} pairs={}".format(
                    name,z["gamma0"],z["condition_number"],z["rank"],z["required_rank"],z["pairs"]
                )
            )
        else:
            print("BRANCH {:30s} FAILED {}".format(name,branch_errors.get(name,"")))

if __name__=="__main__":
    main()
