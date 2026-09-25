import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict, Counter

import joblib
import numpy as np

target_path, stage_map_path, model_path, pred_path, pair_target_path, pair_macro_path, target_macro_path, physical_target_path, config_diag_path, bootstrap_path, summary_path = sys.argv[1:12]

FEATURES = ["Va_EKF_CG","alpha","beta","p","q","r","aileron","elevator","ice_status"]
TARGETS = ["Faero_X","Faero_Y","Faero_Z","Maero_L","Maero_M","Maero_N"]
DEV_SCORE = 0.564390421460009
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
    if "clean" in s: return "clean"
    if "iced" in s or s=="ice" or s.startswith("ice"): return "iced"
    raise RuntimeError("Unrecognized configuration: "+repr(x))

def flight_id(fn):
    m=re.search(r"_ID_(\d+)\.csv$",fn,re.I)
    if not m: raise RuntimeError("Cannot parse flight ID from "+fn)
    return int(m.group(1))

def fnum(x,name,context):
    try: v=float(x)
    except Exception: raise RuntimeError(f"Non-numeric {name} in {context}: {x!r}")
    if not math.isfinite(v): raise RuntimeError(f"Non-finite {name} in {context}")
    return v

def write_csv(path,rows):
    if not rows: raise RuntimeError("Refusing to write empty artifact: "+path)
    fields=[]; seen=set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    with open(path,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

bundle=joblib.load(model_path)
model=bundle["model"]

if bundle["selected_config_id"]!="ET_FULL_LEAF1": raise RuntimeError("Frozen model selected-config drift")
if list(bundle["feature_names"])!=FEATURES: raise RuntimeError("Frozen model feature schema drift")
if list(bundle["target_names"])!=TARGETS: raise RuntimeError("Frozen model target schema drift")
if int(bundle["fit_count"])!=1: raise RuntimeError("Frozen model fit-count drift")

y_mu=np.asarray(bundle["target_weighted_mean"],dtype=float)
y_sd=np.asarray(bundle["target_weighted_sd"],dtype=float)
if y_mu.shape!=(6,) or y_sd.shape!=(6,) or not np.all(np.isfinite(y_mu)) or not np.all(np.isfinite(y_sd)) or np.any(y_sd<=1e-12):
    raise RuntimeError("Invalid frozen development target scaling")

stage_map={}
with open(stage_map_path,"r",encoding="utf-8-sig",newline="") as f:
    for row in csv.DictReader(f):
        cfg=norm_cfg(row["configuration"]); fn=row["file_name"].strip()
        if flight_id(fn)<4: raise RuntimeError("Development file leaked into locked stage map")
        key=(cfg,fn)
        if key in stage_map: raise RuntimeError("Duplicate stage-map key")
        stage_map[key]=row["stage_path"]

if len(stage_map)!=48: raise RuntimeError(f"Expected 48 staged files, observed {len(stage_map)}")

source_rows={}
raw_source_rows_read=0
for key,path in stage_map.items():
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        r=csv.DictReader(f)
        missing=[c for c in FEATURES+["TIME"] if c not in (r.fieldnames or [])]
        if missing: raise RuntimeError(f"Locked source missing fields {missing}: {path}")
        rr=list(r)
    source_rows[key]=rr
    raw_source_rows_read += len(rr)

with open(target_path,"r",encoding="utf-8-sig",newline="") as f:
    r=csv.DictReader(f)
    required=["configuration","file_name","sample_index","TIME"]+TARGETS
    missing=[c for c in required if c not in (r.fieldnames or [])]
    if missing: raise RuntimeError("Locked target artifact missing fields: "+repr(missing))
    target_rows=list(r)

if not target_rows: raise RuntimeError("Locked target artifact is empty")

X=[]; Y=[]; meta=[]; seen=set()
pair_config_counts=defaultdict(Counter)
ice_values=defaultdict(set)
max_time_diff=0.0

for tr in target_rows:
    cfg=norm_cfg(tr["configuration"]); fn=tr["file_name"].strip(); fid=flight_id(fn)
    if fid<4: raise RuntimeError("Development ID found in locked target artifact")
    key=(cfg,fn)
    if key not in source_rows: raise RuntimeError("Locked target row has no staged source mapping: "+repr(key))

    si=int(float(tr["sample_index"]))
    if si<0 or si>=len(source_rows[key]): raise RuntimeError("sample_index out of range")

    sr=source_rows[key][si]
    tt=fnum(tr["TIME"],"TIME","target"); ts=fnum(sr["TIME"],"TIME","source")
    td=abs(tt-ts); max_time_diff=max(max_time_diff,td)
    if td>1e-9: raise RuntimeError(f"TIME identity mismatch >1e-9 for {key} sample {si}")

    ident=(cfg,fn,si)
    if ident in seen: raise RuntimeError("Duplicate locked joined identity")
    seen.add(ident)

    xv=[fnum(sr[c],c,f"source:{key}:{si}") for c in FEATURES]
    yv=[fnum(tr[c],c,f"target:{key}:{si}") for c in TARGETS]

    X.append(xv); Y.append(yv)
    meta.append({"configuration":cfg,"file_name":fn,"sample_index":si,"TIME":tt,"pair_group":fn,"flight_id":fid})
    pair_config_counts[fn][cfg]+=1
    ice_values[cfg].add(xv[-1])

X=np.asarray(X,dtype=float); Y=np.asarray(Y,dtype=float)
if X.shape[1]!=9 or Y.shape[1]!=6: raise RuntimeError("Locked joined feature/target shape drift")
if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)): raise RuntimeError("Non-finite locked joined data")

pairs=sorted(set(m["pair_group"] for m in meta))
if len(pairs)!=24: raise RuntimeError(f"Expected 24 exact pairs, observed {len(pairs)}")
for p in pairs:
    if pair_config_counts[p]["clean"]<=0 or pair_config_counts[p]["iced"]<=0:
        raise RuntimeError("Exact pair lacks clean or iced rows: "+p)

if set(ice_values)!={"clean","iced"}: raise RuntimeError("ice_status config audit incomplete")
if len(ice_values["clean"])!=1 or len(ice_values["iced"])!=1: raise RuntimeError("ice_status not constant within config")
if next(iter(ice_values["clean"]))==next(iter(ice_values["iced"])): raise RuntimeError("ice_status fails clean/iced distinction")

# EXACTLY ONE frozen-model prediction call.
zpred=model.predict(X)
if zpred.shape!=Y.shape: raise RuntimeError("Prediction shape mismatch")
Yhat=zpred*y_sd+y_mu
if not np.all(np.isfinite(Yhat)): raise RuntimeError("Non-finite predictions")

pair_arr=np.asarray([m["pair_group"] for m in meta],dtype=object)
cfg_arr=np.asarray([m["configuration"] for m in meta],dtype=object)

pair_target=[]
for p in pairs:
    mask=(pair_arr==p); err=Yhat[mask]-Y[mask]
    rmse=np.sqrt(np.mean(err*err,axis=0)); mae=np.mean(np.abs(err),axis=0); nrmse=rmse/y_sd
    for j,t in enumerate(TARGETS):
        pair_target.append({
            "pair_group":p,"flight_id":flight_id(p),"target":t,"rows":int(np.sum(mask)),
            "rmse":float(rmse[j]),"mae":float(mae[j]),"development_scale_sd":float(y_sd[j]),"nrmse":float(nrmse[j])
        })

primary=float(np.mean([r["nrmse"] for r in pair_target]))

pair_macro=[]
for p in pairs:
    rr=[r for r in pair_target if r["pair_group"]==p]
    pair_macro.append({
        "pair_group":p,"flight_id":flight_id(p),
        "pair_macro_nrmse":float(np.mean([r["nrmse"] for r in rr])),
        "min_target_nrmse":float(np.min([r["nrmse"] for r in rr])),
        "max_target_nrmse":float(np.max([r["nrmse"] for r in rr]))
    })

target_macro=[]
for t in TARGETS:
    rr=[r for r in pair_target if r["target"]==t]
    vals=np.asarray([r["nrmse"] for r in rr],dtype=float)
    target_macro.append({
        "target":t,"pair_macro_nrmse":float(np.mean(vals)),"pair_nrmse_sd":float(np.std(vals,ddof=1)),
        "pair_nrmse_min":float(np.min(vals)),"pair_nrmse_max":float(np.max(vals)),
        "development_scale_sd":float(y_sd[TARGETS.index(t)])
    })

physical=[]
for j,t in enumerate(TARGETS):
    e=Yhat[:,j]-Y[:,j]
    physical.append({
        "target":t,"rows":len(Y),"rmse":float(np.sqrt(np.mean(e*e))),
        "mae":float(np.mean(np.abs(e))),"mean_signed_error_bias":float(np.mean(e))
    })

config_diag=[]
for cfg in ("clean","iced"):
    mask=(cfg_arr==cfg)
    for j,t in enumerate(TARGETS):
        e=Yhat[mask,j]-Y[mask,j]
        config_diag.append({
            "configuration":cfg,"target":t,"rows":int(np.sum(mask)),
            "rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(np.abs(e))),
            "mean_signed_error_bias":float(np.mean(e))
        })

matrix=np.empty((24,6),dtype=float)
for i,p in enumerate(pairs):
    for j,t in enumerate(TARGETS):
        rr=[r for r in pair_target if r["pair_group"]==p and r["target"]==t]
        if len(rr)!=1: raise RuntimeError("Pair-target matrix construction failure")
        matrix[i,j]=float(rr[0]["nrmse"])

rng=np.random.default_rng(BOOT_SEED)
boot_primary=np.empty(BOOT_N,dtype=float)
boot_target=np.empty((BOOT_N,6),dtype=float)
for b in range(BOOT_N):
    idx=rng.integers(0,24,size=24)
    s=matrix[idx,:]
    boot_primary[b]=float(np.mean(s)); boot_target[b,:]=np.mean(s,axis=0)

boot_rows=[{
    "quantity":"PRIMARY_LOCKED_EXTERNAL_SCORE","estimate":primary,
    "ci_low_2p5":float(np.quantile(boot_primary,0.025)),
    "ci_high_97p5":float(np.quantile(boot_primary,0.975)),
    "bootstrap_replicates":BOOT_N,"bootstrap_seed":BOOT_SEED,"resampling_unit":"EXACT_PAIR"
}]
for j,t in enumerate(TARGETS):
    boot_rows.append({
        "quantity":"TARGET_"+t,"estimate":float(np.mean(matrix[:,j])),
        "ci_low_2p5":float(np.quantile(boot_target[:,j],0.025)),
        "ci_high_97p5":float(np.quantile(boot_target[:,j],0.975)),
        "bootstrap_replicates":BOOT_N,"bootstrap_seed":BOOT_SEED,"resampling_unit":"EXACT_PAIR"
    })

pred_rows=[]
for i,m in enumerate(meta):
    row=dict(m)
    for j,t in enumerate(TARGETS):
        row["true_"+t]=float(Y[i,j]); row["pred_"+t]=float(Yhat[i,j]); row["error_"+t]=float(Yhat[i,j]-Y[i,j])
    pred_rows.append(row)

write_csv(pred_path,pred_rows)
write_csv(pair_target_path,pair_target)
write_csv(pair_macro_path,pair_macro)
write_csv(target_macro_path,target_macro)
write_csv(physical_target_path,physical)
write_csv(config_diag_path,config_diag)
write_csv(bootstrap_path,boot_rows)

ci=boot_rows[0]
summary={
    "execution_integrity":"PASS",
    "scientific_adjudication":"DEFER_TO_P2_A5C",
    "model_bundle_sha256":sha256_file(model_path),
    "locked_target_artifact_sha256":sha256_file(target_path),
    "raw_locked_source_rows_read":int(raw_source_rows_read),
    "locked_target_rows":int(len(target_rows)),
    "locked_joined_rows":int(len(Y)),
    "locked_exact_pairs":24,
    "locked_files":48,
    "locked_flight_ids":sorted(set(int(m["flight_id"]) for m in meta)),
    "max_abs_time_identity_difference":float(max_time_diff),
    "prediction_passes":1,
    "model_predict_calls":1,
    "model_fit_calls":0,
    "model_refit_calls":0,
    "primary_locked_external_score":primary,
    "primary_bootstrap_ci_low_2p5":float(ci["ci_low_2p5"]),
    "primary_bootstrap_ci_high_97p5":float(ci["ci_high_97p5"]),
    "development_outer_mean_reference_score":DEV_SCORE,
    "validation_to_development_score_ratio":primary/DEV_SCORE,
    "validation_minus_development_score":primary-DEV_SCORE,
    "target_macro_nrmse":{r["target"]:r["pair_macro_nrmse"] for r in target_macro},
    "bootstrap_replicates":BOOT_N,
    "bootstrap_seed":BOOT_SEED,
    "binary_success_threshold":"NONE",
    "source_mutation":"NONE_PENDING_POSTCHECK"
}
with open(summary_path,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,separators=(",",":")))
