import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

(
    identity_path,
    stage_map_path,
    support_npz_path,
    model_path,
    row_out,
    coverage_out,
    pair_out,
    estimand_out,
    sensitivity_out,
    bootstrap_out,
    summary_out,
) = sys.argv[1:12]

CONT = ["Va_EKF_CG","alpha","beta","p","q","r","aileron","elevator"]
FEATURES = CONT + ["ice_status"]
TARGETS = ["Faero_X","Faero_Y","Faero_Z","Maero_L","Maero_M","Maero_N"]
BOOT_N = 10000
BOOT_SEED = 20260919

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest().upper()

def norm_cfg(x):
    s=(x or "").strip().lower()
    if "clean" in s:
        return "clean"
    if "iced" in s or s=="ice" or s.startswith("ice"):
        return "iced"
    raise RuntimeError("Unrecognized configuration: "+repr(x))

def flight_id(fn):
    m=re.search(r"_ID_(\d+)\.csv$",fn,re.I)
    if not m:
        raise RuntimeError("Cannot parse flight ID from "+fn)
    return int(m.group(1))

def fnum(x,name,context):
    try:
        v=float(x)
    except Exception:
        raise RuntimeError(f"Non-numeric {name} in {context}: {x!r}")
    if not math.isfinite(v):
        raise RuntimeError(f"Non-finite {name} in {context}")
    return v

def write_csv(path, rows):
    if not rows:
        raise RuntimeError("Refusing to write empty artifact: "+path)
    fields=[]
    seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    with open(path,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

# ----------------------------------------------------------------------------------
# Bind support reference.
# ----------------------------------------------------------------------------------
npz=np.load(support_npz_path)
cont_np=[str(x) for x in npz["continuous_features"].tolist()]
if cont_np!=CONT:
    raise RuntimeError("Support continuous-feature schema drift")

mu=np.asarray(npz["weighted_mean"],dtype=float)
sd=np.asarray(npz["weighted_sd"],dtype=float)
zc=np.asarray(npz["clean_reference_scaled"],dtype=float)
zi=np.asarray(npz["iced_reference_scaled"],dtype=float)
clean_status=float(np.asarray(npz["clean_ice_status_value"],dtype=float).ravel()[0])
iced_status=float(np.asarray(npz["iced_ice_status_value"],dtype=float).ravel()[0])
k=int(np.asarray(npz["k"],dtype=int).ravel()[0])
q90=float(np.asarray(npz["q90"],dtype=float).ravel()[0])
q95=float(np.asarray(npz["q95"],dtype=float).ravel()[0])
q99=float(np.asarray(npz["q99"],dtype=float).ravel()[0])

if mu.shape!=(8,) or sd.shape!=(8,) or np.any(sd<=1e-12):
    raise RuntimeError("Invalid support scaling")
if not (q90 <= q95 <= q99):
    raise RuntimeError("Support-threshold ordering drift")
if k!=10:
    raise RuntimeError("kNN k drift")

# ----------------------------------------------------------------------------------
# Bind model.
# ----------------------------------------------------------------------------------
bundle=joblib.load(model_path)
model=bundle["model"]

if bundle["selected_config_id"]!="ET_FULL_LEAF1":
    raise RuntimeError("Frozen model config drift")
if list(bundle["feature_names"])!=FEATURES:
    raise RuntimeError("Frozen model feature schema drift")
if list(bundle["target_names"])!=TARGETS:
    raise RuntimeError("Frozen model target schema drift")
if int(bundle["fit_count"])!=1:
    raise RuntimeError("Frozen model fit-count drift")

y_mu=np.asarray(bundle["target_weighted_mean"],dtype=float)
y_sd=np.asarray(bundle["target_weighted_sd"],dtype=float)
if y_mu.shape!=(6,) or y_sd.shape!=(6,) or np.any(y_sd<=1e-12):
    raise RuntimeError("Frozen target scaling drift")

# ----------------------------------------------------------------------------------
# Stage map.
# ----------------------------------------------------------------------------------
stage_map={}
with open(stage_map_path,"r",encoding="utf-8-sig",newline="") as f:
    for row in csv.DictReader(f):
        cfg=norm_cfg(row["configuration"])
        fn=row["file_name"].strip()
        if flight_id(fn)<4:
            raise RuntimeError("Development file leaked into locked stage")
        key=(cfg,fn)
        if key in stage_map:
            raise RuntimeError("Duplicate stage-map identity")
        stage_map[key]=row["stage_path"]

if len(stage_map)!=48:
    raise RuntimeError(f"Expected 48 staged files, observed {len(stage_map)}")

# Cache raw locked source rows.
source_rows={}
for key,path in stage_map.items():
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        r=csv.DictReader(f)
        missing=[c for c in FEATURES+["TIME"] if c not in (r.fieldnames or [])]
        if missing:
            raise RuntimeError(f"Source missing {missing}: {path}")
        source_rows[key]=list(r)

# ----------------------------------------------------------------------------------
# Exact 20,537-row locked identity cohort from A5B-R1.
# ----------------------------------------------------------------------------------
with open(identity_path,"r",encoding="utf-8-sig",newline="") as f:
    r=csv.DictReader(f)
    required=["configuration","file_name","sample_index","TIME","pair_group","flight_id"]
    missing=[c for c in required if c not in (r.fieldnames or [])]
    if missing:
        raise RuntimeError("Locked identity artifact missing fields: "+repr(missing))
    ids=list(r)

if len(ids)!=20537:
    raise RuntimeError(f"Expected 20537 locked identities, observed {len(ids)}")

Xcont=[]
observed_cfg=[]
pair_group=[]
meta=[]
seen=set()
max_time_diff=0.0

for ir in ids:
    cfg=norm_cfg(ir["configuration"])
    fn=ir["file_name"].strip()
    fid=int(ir["flight_id"])
    if fid<4 or fid!=flight_id(fn):
        raise RuntimeError("Locked flight-ID mismatch")

    si=int(float(ir["sample_index"]))
    key=(cfg,fn)
    if key not in source_rows:
        raise RuntimeError("Identity has no locked staged source: "+repr(key))
    if si<0 or si>=len(source_rows[key]):
        raise RuntimeError("sample_index out of range")

    sr=source_rows[key][si]
    tt=fnum(ir["TIME"],"TIME","identity")
    ts=fnum(sr["TIME"],"TIME","source")
    td=abs(tt-ts)
    max_time_diff=max(max_time_diff,td)
    if td>1e-9:
        raise RuntimeError("TIME identity mismatch")

    ident=(cfg,fn,si)
    if ident in seen:
        raise RuntimeError("Duplicate locked identity")
    seen.add(ident)

    xv=[fnum(sr[c],c,f"{key}:{si}") for c in CONT]
    Xcont.append(xv)
    observed_cfg.append(cfg)
    pair_group.append(ir["pair_group"])
    meta.append({
        "configuration":cfg,
        "file_name":fn,
        "sample_index":si,
        "TIME":tt,
        "pair_group":ir["pair_group"],
        "flight_id":fid,
    })

Xcont=np.asarray(Xcont,dtype=float)
observed_cfg=np.asarray(observed_cfg,dtype=object)
pair_group=np.asarray(pair_group,dtype=object)

if Xcont.shape!=(20537,8) or not np.all(np.isfinite(Xcont)):
    raise RuntimeError("Locked continuous-feature matrix drift")

pairs=sorted(set(pair_group.tolist()))
if len(pairs)!=24:
    raise RuntimeError(f"Expected 24 exact pairs, observed {len(pairs)}")

# ----------------------------------------------------------------------------------
# Frozen common-support geometry.
# ----------------------------------------------------------------------------------
Z=(Xcont-mu)/sd
nn_c=NearestNeighbors(n_neighbors=k,algorithm="auto",n_jobs=2).fit(zc)
nn_i=NearestNeighbors(n_neighbors=k,algorithm="auto",n_jobs=2).fit(zi)

d_clean=np.mean(nn_c.kneighbors(Z,return_distance=True)[0],axis=1)
d_iced=np.mean(nn_i.kneighbors(Z,return_distance=True)[0],axis=1)

support90=(d_clean<=q90)&(d_iced<=q90)
support95=(d_clean<=q95)&(d_iced<=q95)
support99=(d_clean<=q99)&(d_iced<=q99)

if np.any(support90 & ~support95) or np.any(support95 & ~support99):
    raise RuntimeError("Support nesting failure")
if int(np.sum(support95))==0:
    raise RuntimeError("Primary q95 common support is empty")

# q99 is the union needed for the predeclared q90/q95/q99 sensitivity analysis.
idx99=np.where(support99)[0]
Xbase=Xcont[idx99,:]

Xclean=np.column_stack([Xbase, np.full(len(idx99),clean_status,dtype=float)])
Xiced=np.column_stack([Xbase, np.full(len(idx99),iced_status,dtype=float)])
Xstack=np.vstack([Xclean,Xiced])

# EXACTLY ONE model.predict call in P2-A6B.
zpred=model.predict(Xstack)
if zpred.shape!=(2*len(idx99),6):
    raise RuntimeError("Counterfactual prediction shape drift")

Yhat=zpred*y_sd+y_mu
Yc=Yhat[:len(idx99),:]
Yi=Yhat[len(idx99):,:]
Delta=Yi-Yc

if not np.all(np.isfinite(Yc)) or not np.all(np.isfinite(Yi)) or not np.all(np.isfinite(Delta)):
    raise RuntimeError("Non-finite counterfactual predictions")

alpha=Xbase[:,1]
ca=np.cos(alpha); sa=np.sin(alpha)

Dc=-(Yc[:,0]*ca + Yc[:,2]*sa)
Di=-(Yi[:,0]*ca + Yi[:,2]*sa)
Lc=Yc[:,0]*sa - Yc[:,2]*ca
Li=Yi[:,0]*sa - Yi[:,2]*ca

deltaD=Di-Dc
deltaL=Li-Lc

metric_names=[
    "DeltaD_cf",
    "DeltaLift_cf",
    "DeltaFaero_X",
    "DeltaFaero_Y",
    "DeltaFaero_Z",
    "DeltaMaero_L",
    "DeltaMaero_M",
    "DeltaMaero_N",
]
metric_arrays=[
    deltaD,
    deltaL,
    Delta[:,0],
    Delta[:,1],
    Delta[:,2],
    Delta[:,3],
    Delta[:,4],
    Delta[:,5],
]
metric_units=["N","N","N","N","N","N m","N m","N m"]

# map global row index -> q99-local prediction index
local_of_global={int(g):i for i,g in enumerate(idx99.tolist())}

coverage_rows=[]
for threshold_name,mask,thr in [
    ("Q90",support90,q90),
    ("Q95_PRIMARY",support95,q95),
    ("Q99",support99,q99),
]:
    coverage_rows.append({
        "support_rule":threshold_name,
        "scope":"OVERALL",
        "group":"ALL",
        "threshold":thr,
        "retained_rows":int(np.sum(mask)),
        "total_rows":len(mask),
        "coverage_fraction":float(np.mean(mask)),
    })
    for cfg in ("clean","iced"):
        m=(observed_cfg==cfg)
        coverage_rows.append({
            "support_rule":threshold_name,
            "scope":"OBSERVED_CONFIGURATION",
            "group":cfg,
            "threshold":thr,
            "retained_rows":int(np.sum(mask&m)),
            "total_rows":int(np.sum(m)),
            "coverage_fraction":float(np.mean(mask[m])),
        })
    for p in pairs:
        m=(pair_group==p)
        coverage_rows.append({
            "support_rule":threshold_name,
            "scope":"EXACT_PAIR",
            "group":p,
            "threshold":thr,
            "retained_rows":int(np.sum(mask&m)),
            "total_rows":int(np.sum(m)),
            "coverage_fraction":float(np.mean(mask[m])),
        })

row_results=[]
for gidx in idx99.tolist():
    li=local_of_global[int(gidx)]
    r=dict(meta[gidx])
    r.update({
        "d_clean":float(d_clean[gidx]),
        "d_iced":float(d_iced[gidx]),
        "support_q90":bool(support90[gidx]),
        "support_q95_primary":bool(support95[gidx]),
        "support_q99":bool(support99[gidx]),
        "ice_status_clean_query":clean_status,
        "ice_status_iced_query":iced_status,
        "D_clean_status":float(Dc[li]),
        "D_iced_status":float(Di[li]),
        "DeltaD_cf":float(deltaD[li]),
        "Lift_clean_status":float(Lc[li]),
        "Lift_iced_status":float(Li[li]),
        "DeltaLift_cf":float(deltaL[li]),
    })
    for j,t in enumerate(TARGETS):
        r["pred_clean_"+t]=float(Yc[li,j])
        r["pred_iced_"+t]=float(Yi[li,j])
        r["Delta"+t]=float(Delta[li,j])
    row_results.append(r)

def analyze_threshold(name, mask):
    gpairs=[p for p in pairs if np.any(mask & (pair_group==p))]
    if len(gpairs)<2:
        raise RuntimeError(f"{name}: fewer than 2 supported exact pairs")

    pair_rows=[]
    matrix=np.empty((len(gpairs),len(metric_names)),dtype=float)

    for pi,p in enumerate(gpairs):
        global_idx=np.where(mask & (pair_group==p))[0]
        local_idx=np.asarray([local_of_global[int(g)] for g in global_idx],dtype=int)

        row={
            "support_rule":name,
            "pair_group":p,
            "flight_id":flight_id(p),
            "supported_rows":len(global_idx),
        }
        for j,(mn,arr) in enumerate(zip(metric_names,metric_arrays)):
            val=float(np.mean(arr[local_idx]))
            row["pair_mean_"+mn]=val
            matrix[pi,j]=val
        pair_rows.append(row)

    point=np.mean(matrix,axis=0)
    rng=np.random.default_rng(BOOT_SEED)
    boot=np.empty((BOOT_N,len(metric_names)),dtype=float)
    n=len(gpairs)
    for b in range(BOOT_N):
        idx=rng.integers(0,n,size=n)
        boot[b,:]=np.mean(matrix[idx,:],axis=0)

    est_rows=[]
    boot_rows=[]
    for j,(mn,unit) in enumerate(zip(metric_names,metric_units)):
        lo=float(np.quantile(boot[:,j],0.025))
        hi=float(np.quantile(boot[:,j],0.975))
        est_rows.append({
            "support_rule":name,
            "estimand":mn,
            "unit":unit,
            "pair_macro_mean":float(point[j]),
            "supported_pairs":n,
            "supported_rows":int(np.sum(mask)),
        })
        boot_rows.append({
            "support_rule":name,
            "estimand":mn,
            "unit":unit,
            "estimate":float(point[j]),
            "ci_low_2p5":lo,
            "ci_high_97p5":hi,
            "bootstrap_replicates":BOOT_N,
            "bootstrap_seed":BOOT_SEED,
            "resampling_unit":"EXACT_PAIR",
        })

    drag_pair_means=matrix[:,0]
    drag_est=float(point[0])
    drag_lo=float(np.quantile(boot[:,0],0.025))
    drag_hi=float(np.quantile(boot[:,0],0.975))
    positive_pairs=int(np.sum(drag_pair_means>0))

    if drag_lo>0:
        drag_class="CONSISTENT_WITH_P1_POSITIVE_DRAG_DIRECTION"
    elif drag_hi<0:
        drag_class="INCONSISTENT_WITH_P1_POSITIVE_DRAG_DIRECTION"
    else:
        drag_class="INCONCLUSIVE_RELATIVE_TO_P1_POSITIVE_DRAG_DIRECTION"

    sensitivity={
        "support_rule":name,
        "supported_rows":int(np.sum(mask)),
        "supported_pairs":n,
        "pair_macro_mean_DeltaD_cf_N":drag_est,
        "ci_low_2p5_N":drag_lo,
        "ci_high_97p5_N":drag_hi,
        "positive_pair_means":positive_pairs,
        "positive_pair_fraction":float(positive_pairs/n),
        "direction_classification":drag_class,
    }
    return pair_rows,est_rows,boot_rows,sensitivity

all_pair=[]
all_est=[]
all_boot=[]
sensitivity=[]

for name,mask in [
    ("Q90",support90),
    ("Q95_PRIMARY",support95),
    ("Q99",support99),
]:
    pr,er,br,sr=analyze_threshold(name,mask)
    all_pair.extend(pr)
    all_est.extend(er)
    all_boot.extend(br)
    sensitivity.append(sr)

primary=[x for x in sensitivity if x["support_rule"]=="Q95_PRIMARY"][0]

write_csv(row_out,row_results)
write_csv(coverage_out,coverage_rows)
write_csv(pair_out,all_pair)
write_csv(estimand_out,all_est)
write_csv(sensitivity_out,sensitivity)
write_csv(bootstrap_out,all_boot)

summary={
    "execution_integrity":"PASS",
    "scientific_adjudication":"DEFER_TERMINAL_CLAIM_FREEZE_TO_P2_A6C",
    "identity_rows":len(ids),
    "exact_pairs":len(pairs),
    "support_thresholds":{"q90":q90,"q95":q95,"q99":q99},
    "support_rows":{
        "q90":int(np.sum(support90)),
        "q95":int(np.sum(support95)),
        "q99":int(np.sum(support99)),
    },
    "support_coverage":{
        "q90":float(np.mean(support90)),
        "q95":float(np.mean(support95)),
        "q99":float(np.mean(support99)),
    },
    "q95_supported_pairs":int(primary["supported_pairs"]),
    "counterfactual_query_states_predicted":int(len(idx99)),
    "stacked_model_prediction_rows":int(2*len(idx99)),
    "model_fit_calls":0,
    "model_refit_calls":0,
    "model_predict_calls":1,
    "primary_estimand":"PAIR-MACRO MEAN DeltaD_cf [N] ON Q95 BOTH-CONFIG COMMON SUPPORT",
    "primary_DeltaD_cf_N":float(primary["pair_macro_mean_DeltaD_cf_N"]),
    "primary_ci_low_2p5_N":float(primary["ci_low_2p5_N"]),
    "primary_ci_high_97p5_N":float(primary["ci_high_97p5_N"]),
    "primary_positive_pair_means":int(primary["positive_pair_means"]),
    "primary_positive_pair_fraction":float(primary["positive_pair_fraction"]),
    "prospective_P1_drag_direction_result":primary["direction_classification"],
    "bootstrap_replicates":BOOT_N,
    "bootstrap_seed":BOOT_SEED,
    "causal_effect_claim":"PROHIBITED",
    "derivative_consistency_claim":"NOT_TESTED",
    "model_sha256":sha256_file(model_path),
    "support_reference_sha256":sha256_file(support_npz_path),
    "max_abs_time_identity_difference":float(max_time_diff),
}

with open(summary_out,"w",encoding="utf-8") as f:
    json.dump(summary,f,indent=2)

print(json.dumps(summary,separators=(",",":")))
