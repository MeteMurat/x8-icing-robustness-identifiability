import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

(
    locked_predictions_path,
    pair_target_errors_path,
    stage_map_path,
    support_npz_path,
    model_path,
    perm_repeat_out,
    perm_pair_out,
    perm_importance_out,
    native_out,
    concordance_out,
    dispersion_pair_target_out,
    dispersion_assoc_out,
    support_pair_out,
    support_assoc_out,
    summary_out,
) = sys.argv[1:16]

CONT = ["Va_EKF_CG","alpha","beta","p","q","r","aileron","elevator"]
FEATURES = CONT + ["ice_status"]
TARGETS = ["Faero_X","Faero_Y","Faero_Z","Maero_L","Maero_M","Maero_N"]

PERM_REPEATS = 20
SEED = 20260919
BOOT_N = 10000
TREE_COUNT = 256
EXPECTED_BASELINE = 0.5399890077502464

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

def write_csv(path,rows):
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

def spearman(x,y):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    if len(x)<2 or len(y)<2:
        return float("nan")
    rx=rankdata(x,method="average")
    ry=rankdata(y,method="average")
    sx=float(np.std(rx))
    sy=float(np.std(ry))
    if sx<=0 or sy<=0:
        return float("nan")
    return float(np.corrcoef(rx,ry)[0,1])

def bootstrap_spearman(x,y,seed):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    n=len(x)
    rng=np.random.default_rng(seed)
    vals=np.empty(BOOT_N,dtype=float)
    vals.fill(np.nan)
    for b in range(BOOT_N):
        idx=rng.integers(0,n,size=n)
        vals[b]=spearman(x[idx],y[idx])
    finite=vals[np.isfinite(vals)]
    if len(finite)<int(0.95*BOOT_N):
        raise RuntimeError("Too few finite bootstrap Spearman replicates")
    return (
        float(np.quantile(finite,0.025)),
        float(np.quantile(finite,0.975)),
        int(len(finite)),
    )

def classify_ci(lo,hi):
    if lo>0:
        return "POSITIVE_ASSOCIATION"
    if hi<0:
        return "INVERSE_ASSOCIATION"
    return "INCONCLUSIVE"

# --------------------------------------------------------------------------------------------------
# Frozen model.
# --------------------------------------------------------------------------------------------------
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
if len(model.estimators_)!=TREE_COUNT:
    raise RuntimeError("Frozen model tree-count drift")

y_sd=np.asarray(bundle["target_weighted_sd"],dtype=float)
if y_sd.shape!=(6,) or np.any(y_sd<=1e-12):
    raise RuntimeError("Frozen target-scale drift")

# --------------------------------------------------------------------------------------------------
# Locked identities, true targets, frozen baseline predictions.
# --------------------------------------------------------------------------------------------------
with open(locked_predictions_path,"r",encoding="utf-8-sig",newline="") as f:
    r=csv.DictReader(f)
    required=["configuration","file_name","sample_index","TIME","pair_group","flight_id"]
    for t in TARGETS:
        required += ["true_"+t,"pred_"+t]
    missing=[c for c in required if c not in (r.fieldnames or [])]
    if missing:
        raise RuntimeError("Locked prediction artifact missing fields: "+repr(missing))
    ids=list(r)

if len(ids)!=20537:
    raise RuntimeError(f"Expected 20537 locked rows, observed {len(ids)}")

cfg=np.asarray([norm_cfg(r["configuration"]) for r in ids],dtype=object)
pair=np.asarray([r["pair_group"] for r in ids],dtype=object)
file_name=np.asarray([r["file_name"].strip() for r in ids],dtype=object)
sample_index=np.asarray([int(float(r["sample_index"])) for r in ids],dtype=int)
time=np.asarray([fnum(r["TIME"],"TIME","locked_predictions") for r in ids],dtype=float)
flight=np.asarray([int(r["flight_id"]) for r in ids],dtype=int)
Y=np.asarray([[fnum(r["true_"+t],"true_"+t,"locked_predictions") for t in TARGETS] for r in ids],dtype=float)
Yhat0=np.asarray([[fnum(r["pred_"+t],"pred_"+t,"locked_predictions") for t in TARGETS] for r in ids],dtype=float)

if sorted(set(flight.tolist()))[0] < 4:
    raise RuntimeError("Development ID leaked into locked predictions")

pairs=sorted(set(pair.tolist()))
if len(pairs)!=24:
    raise RuntimeError(f"Expected 24 locked exact pairs, observed {len(pairs)}")

# --------------------------------------------------------------------------------------------------
# Frozen pair-target baseline errors.
# --------------------------------------------------------------------------------------------------
baseline_pair_target={}
with open(pair_target_errors_path,"r",encoding="utf-8-sig",newline="") as f:
    rows=list(csv.DictReader(f))

if len(rows)!=24*6:
    raise RuntimeError(f"Expected 144 pair-target baseline rows, observed {len(rows)}")

for r in rows:
    key=(r["pair_group"],r["target"])
    if key in baseline_pair_target:
        raise RuntimeError("Duplicate baseline pair-target error")
    baseline_pair_target[key]=float(r["nrmse"])

baseline_pair_macro={}
for p in pairs:
    vals=[baseline_pair_target[(p,t)] for t in TARGETS]
    baseline_pair_macro[p]=float(np.mean(vals))

baseline_primary=float(np.mean([baseline_pair_macro[p] for p in pairs]))
if abs(baseline_primary-EXPECTED_BASELINE)>1e-12:
    raise RuntimeError(f"Frozen baseline primary-score mismatch: {baseline_primary}")

# --------------------------------------------------------------------------------------------------
# Reconstruct exact frozen feature matrix from staged source rows.
# --------------------------------------------------------------------------------------------------
stage_map={}
with open(stage_map_path,"r",encoding="utf-8-sig",newline="") as f:
    for r in csv.DictReader(f):
        key=(norm_cfg(r["configuration"]),r["file_name"].strip())
        if key in stage_map:
            raise RuntimeError("Duplicate locked stage-map key")
        stage_map[key]=r["stage_path"]

if len(stage_map)!=48:
    raise RuntimeError("Expected 48 locked staged files")

source_rows={}
for key,path in stage_map.items():
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        rr=csv.DictReader(f)
        missing=[c for c in FEATURES+["TIME"] if c not in (rr.fieldnames or [])]
        if missing:
            raise RuntimeError(f"Locked source missing fields {missing}: {path}")
        source_rows[key]=list(rr)

X=[]
max_time_diff=0.0
seen=set()

for i in range(len(ids)):
    key=(cfg[i],file_name[i])
    if key not in source_rows:
        raise RuntimeError("Locked identity has no staged source mapping")
    si=int(sample_index[i])
    if si<0 or si>=len(source_rows[key]):
        raise RuntimeError("Locked sample_index out of range")

    sr=source_rows[key][si]
    td=abs(time[i]-fnum(sr["TIME"],"TIME","staged_source"))
    max_time_diff=max(max_time_diff,td)
    if td>1e-9:
        raise RuntimeError("Locked TIME identity mismatch")

    ident=(cfg[i],file_name[i],si)
    if ident in seen:
        raise RuntimeError("Duplicate locked identity")
    seen.add(ident)

    X.append([fnum(sr[c],c,"staged_source") for c in FEATURES])

X=np.asarray(X,dtype=float)
if X.shape!=(20537,9) or not np.all(np.isfinite(X)):
    raise RuntimeError("Frozen feature matrix drift")

# Verify unpermuted frozen predictions are consistent with frozen model.
# This is NOT recomputed with model.predict: artifact identity is trusted from P2-A5B-R1.
if not np.all(np.isfinite(Yhat0)):
    raise RuntimeError("Non-finite frozen baseline predictions")

# --------------------------------------------------------------------------------------------------
# Group-preserving permutation importance.
# --------------------------------------------------------------------------------------------------
strata=[]
for p in pairs:
    for c in ("clean","iced"):
        idx=np.where((pair==p)&(cfg==c))[0]
        if len(idx)==0:
            raise RuntimeError(f"Empty permutation stratum: {p} / {c}")
        strata.append(idx)

rng=np.random.default_rng(SEED)
perm_repeat_rows=[]
pair_feature_repeat=defaultdict(list)
model_predict_calls=0

for rep in range(PERM_REPEATS):
    print(f"[P2-A7B] permutation repeat {rep+1:02d}/{PERM_REPEATS}",flush=True)
    blocks=[]

    for fk,feature in enumerate(CONT):
        Xp=X.copy()
        for idx in strata:
            perm_idx=rng.permutation(idx)
            Xp[idx,fk]=X[perm_idx,fk]
        blocks.append(Xp)

    Xstack=np.vstack(blocks)

    # One frozen-model prediction call per repeat, containing all eight single-feature permutations.
    zpred=model.predict(Xstack)
    model_predict_calls += 1

    if zpred.shape!=(len(CONT)*len(X),6):
        raise RuntimeError("Permutation prediction shape drift")

    y_mu=np.asarray(bundle["target_weighted_mean"],dtype=float)
    Ystack=zpred*y_sd+y_mu

    for fk,feature in enumerate(CONT):
        yp=Ystack[fk*len(X):(fk+1)*len(X),:]
        pair_macro_perm={}

        for p in pairs:
            m=(pair==p)
            err=yp[m,:]-Y[m,:]
            rmse=np.sqrt(np.mean(err*err,axis=0))
            nrmse=rmse/y_sd
            pair_macro=float(np.mean(nrmse))
            pair_macro_perm[p]=pair_macro
            pair_feature_repeat[(p,feature)].append(pair_macro-baseline_pair_macro[p])

        perm_primary=float(np.mean([pair_macro_perm[p] for p in pairs]))
        perm_repeat_rows.append({
            "repeat":rep+1,
            "feature":feature,
            "baseline_primary_nrmse":baseline_primary,
            "permuted_primary_nrmse":perm_primary,
            "delta_primary_nrmse":perm_primary-baseline_primary,
        })

if model_predict_calls!=PERM_REPEATS:
    raise RuntimeError("Permutation model.predict call-count drift")

perm_pair_rows=[]
for p in pairs:
    for feature in CONT:
        vals=np.asarray(pair_feature_repeat[(p,feature)],dtype=float)
        if len(vals)!=PERM_REPEATS:
            raise RuntimeError("Pair-feature permutation repeat-count drift")
        perm_pair_rows.append({
            "pair_group":p,
            "flight_id":flight_id(p),
            "feature":feature,
            "mean_delta_nrmse_over_repeats":float(np.mean(vals)),
            "sd_delta_nrmse_over_repeats":float(np.std(vals,ddof=1)),
            "min_delta_nrmse_over_repeats":float(np.min(vals)),
            "max_delta_nrmse_over_repeats":float(np.max(vals)),
            "repeats":PERM_REPEATS,
        })

perm_importance_rows=[]
rng_boot_perm=np.random.default_rng(SEED)

for feature in CONT:
    vals=np.asarray([
        r["mean_delta_nrmse_over_repeats"]
        for r in perm_pair_rows
        if r["feature"]==feature
    ],dtype=float)

    if len(vals)!=24:
        raise RuntimeError("Expected 24 pair-level permutation deltas")

    estimate=float(np.mean(vals))
    boot=np.empty(BOOT_N,dtype=float)
    for b in range(BOOT_N):
        idx=rng_boot_perm.integers(0,len(vals),size=len(vals))
        boot[b]=float(np.mean(vals[idx]))

    lo=float(np.quantile(boot,0.025))
    hi=float(np.quantile(boot,0.975))

    if lo>0:
        cls="SUPPORTED_PREDICTIVE_DEPENDENCY"
    elif hi<0:
        cls="NEGATIVE_OR_REDUNDANT"
    else:
        cls="INCONCLUSIVE"

    perm_importance_rows.append({
        "feature":feature,
        "pair_macro_mean_delta_nrmse":estimate,
        "ci_low_2p5":lo,
        "ci_high_97p5":hi,
        "classification":cls,
        "pair_count":24,
        "permutation_repeats":PERM_REPEATS,
        "bootstrap_replicates":BOOT_N,
        "bootstrap_seed":SEED,
    })

perm_importance_rows=sorted(
    perm_importance_rows,
    key=lambda r:r["pair_macro_mean_delta_nrmse"],
    reverse=True
)
for i,r in enumerate(perm_importance_rows,1):
    r["permutation_rank"]=i

# --------------------------------------------------------------------------------------------------
# Native ExtraTrees impurity importance.
# --------------------------------------------------------------------------------------------------
native=np.asarray(model.feature_importances_,dtype=float)
if native.shape!=(9,) or not np.all(np.isfinite(native)):
    raise RuntimeError("Native feature-importance shape drift")
if abs(float(np.sum(native))-1.0)>1e-10:
    raise RuntimeError("Native feature importances do not sum to one")

native_rows=[]
order=np.argsort(-native)
native_rank=np.empty(9,dtype=int)
for rank,idx in enumerate(order,1):
    native_rank[idx]=rank

for j,f in enumerate(FEATURES):
    native_rows.append({
        "feature":f,
        "native_mdi_importance":float(native[j]),
        "native_rank_all_9":int(native_rank[j]),
        "role":"SECONDARY_DESCRIPTIVE_ONLY",
    })

perm_map={r["feature"]:r for r in perm_importance_rows}
native_cont=np.asarray([native[FEATURES.index(f)] for f in CONT],dtype=float)
perm_cont=np.asarray([perm_map[f]["pair_macro_mean_delta_nrmse"] for f in CONT],dtype=float)
rho_concordance=spearman(native_cont,perm_cont)

concordance={
    "continuous_features":CONT,
    "spearman_rho_native_vs_permutation":rho_concordance,
    "significance_test":"NOT_PERFORMED",
    "interpretation":"DESCRIPTIVE_RANK_CONCORDANCE_ONLY",
}

# --------------------------------------------------------------------------------------------------
# Tree-ensemble dispersion diagnostic in standardized target space.
# --------------------------------------------------------------------------------------------------
print("[P2-A7B] tree-ensemble dispersion 256 trees",flush=True)

sum_pred=np.zeros((len(X),6),dtype=np.float64)
sum_sq=np.zeros((len(X),6),dtype=np.float64)
tree_predict_calls=0

for ti,tree in enumerate(model.estimators_,1):
    z=np.asarray(tree.predict(X),dtype=np.float64)
    if z.shape!=(len(X),6):
        raise RuntimeError("Tree prediction shape drift")
    sum_pred += z
    sum_sq += z*z
    tree_predict_calls += 1

    if ti in (1,32,64,96,128,160,192,224,256):
        print(f"[P2-A7B] tree dispersion progress {ti}/256",flush=True)

mean_tree=sum_pred/TREE_COUNT
var_tree=np.maximum(sum_sq/TREE_COUNT - mean_tree*mean_tree,0.0)
tree_sd=np.sqrt(var_tree)

if tree_predict_calls!=TREE_COUNT:
    raise RuntimeError("Tree predict-call count drift")
if not np.all(np.isfinite(tree_sd)):
    raise RuntimeError("Non-finite tree dispersion")

dispersion_pair_target_rows=[]
dispersion_assoc_rows=[]

for tj,t in enumerate(TARGETS):
    U=[]
    E=[]
    P=[]

    for p in pairs:
        m=(pair==p)
        u=float(np.sqrt(np.mean(tree_sd[m,tj]**2)))
        e=float(baseline_pair_target[(p,t)])

        dispersion_pair_target_rows.append({
            "pair_group":p,
            "flight_id":flight_id(p),
            "target":t,
            "tree_dispersion_rms_standardized":u,
            "locked_pair_target_nrmse":e,
        })

        U.append(u); E.append(e); P.append(p)

    rho=spearman(U,E)
    lo,hi,nfinite=bootstrap_spearman(U,E,SEED+100+tj)
    cls=classify_ci(lo,hi)

    dispersion_assoc_rows.append({
        "target":t,
        "spearman_rho":rho,
        "ci_low_2p5":lo,
        "ci_high_97p5":hi,
        "classification":cls,
        "exact_pairs":24,
        "finite_bootstrap_replicates":nfinite,
        "bootstrap_replicates_requested":BOOT_N,
        "bootstrap_seed":SEED+100+tj,
        "uncertainty_role":"ENSEMBLE_DISAGREEMENT_DIAGNOSTIC_ONLY",
    })

# --------------------------------------------------------------------------------------------------
# Support-distance / pair-error diagnostic.
# --------------------------------------------------------------------------------------------------
print("[P2-A7B] support-distance/error diagnostic",flush=True)

npz=np.load(support_npz_path)
cont_np=[str(x) for x in npz["continuous_features"].tolist()]
if cont_np!=CONT:
    raise RuntimeError("Support feature schema drift")

mu=np.asarray(npz["weighted_mean"],dtype=float)
sd=np.asarray(npz["weighted_sd"],dtype=float)
zc=np.asarray(npz["clean_reference_scaled"],dtype=float)
zi=np.asarray(npz["iced_reference_scaled"],dtype=float)
k=int(np.asarray(npz["k"],dtype=int).ravel()[0])

if mu.shape!=(8,) or sd.shape!=(8,) or np.any(sd<=1e-12) or k!=10:
    raise RuntimeError("Support reference drift")

Z=(X[:,:8]-mu)/sd
nn_c=NearestNeighbors(n_neighbors=k,algorithm="auto",n_jobs=2).fit(zc)
nn_i=NearestNeighbors(n_neighbors=k,algorithm="auto",n_jobs=2).fit(zi)

d_clean=np.mean(nn_c.kneighbors(Z,return_distance=True)[0],axis=1)
d_iced=np.mean(nn_i.kneighbors(Z,return_distance=True)[0],axis=1)
support_distance=np.maximum(d_clean,d_iced)

support_pair_rows=[]
S=[]
Epair=[]

for p in pairs:
    m=(pair==p)
    s=float(np.median(support_distance[m]))
    e=float(baseline_pair_macro[p])

    support_pair_rows.append({
        "pair_group":p,
        "flight_id":flight_id(p),
        "median_max_clean_iced_knn_distance":s,
        "locked_pair_macro_nrmse":e,
    })

    S.append(s); Epair.append(e)

rho_support=spearman(S,Epair)
lo_support,hi_support,nfinite_support=bootstrap_spearman(S,Epair,SEED+900)

support_class=classify_ci(lo_support,hi_support)
support_assoc={
    "spearman_rho":rho_support,
    "ci_low_2p5":lo_support,
    "ci_high_97p5":hi_support,
    "classification":support_class,
    "exact_pairs":24,
    "finite_bootstrap_replicates":nfinite_support,
    "bootstrap_replicates_requested":BOOT_N,
    "bootstrap_seed":SEED+900,
    "interpretation":"VALIDITY_DIAGNOSTIC_ONLY_NOT_UNIVERSAL_DOMAIN_GUARANTEE",
}

# --------------------------------------------------------------------------------------------------
# Write artifacts.
# --------------------------------------------------------------------------------------------------
write_csv(perm_repeat_out,perm_repeat_rows)
write_csv(perm_pair_out,perm_pair_rows)
write_csv(perm_importance_out,perm_importance_rows)
write_csv(native_out,native_rows)
write_csv(dispersion_pair_target_out,dispersion_pair_target_rows)
write_csv(dispersion_assoc_out,dispersion_assoc_rows)
write_csv(support_pair_out,support_pair_rows)

with open(concordance_out,"w",encoding="utf-8") as f:
    json.dump(concordance,f,indent=2)

with open(support_assoc_out,"w",encoding="utf-8") as f:
    json.dump(support_assoc,f,indent=2)

summary={
    "execution_integrity":"PASS",
    "scientific_adjudication":"DEFER_TERMINAL_CLAIM_FREEZE_TO_P2_A7C",
    "locked_rows":len(X),
    "locked_exact_pairs":len(pairs),
    "baseline_primary_nrmse":baseline_primary,

    "permutation":{
        "features_ranked":CONT,
        "repeats":PERM_REPEATS,
        "seed":SEED,
        "model_predict_calls":model_predict_calls,
        "top_feature":perm_importance_rows[0]["feature"],
        "top_feature_delta_nrmse":perm_importance_rows[0]["pair_macro_mean_delta_nrmse"],
        "top_feature_ci":[
            perm_importance_rows[0]["ci_low_2p5"],
            perm_importance_rows[0]["ci_high_97p5"],
        ],
        "supported_dependency_count":sum(r["classification"]=="SUPPORTED_PREDICTIVE_DEPENDENCY" for r in perm_importance_rows),
        "inconclusive_count":sum(r["classification"]=="INCONCLUSIVE" for r in perm_importance_rows),
        "negative_or_redundant_count":sum(r["classification"]=="NEGATIVE_OR_REDUNDANT" for r in perm_importance_rows),
    },

    "native_importance":{
        "top_feature_all_9":native_rows[int(np.argmax(native))]["feature"],
        "native_vs_permutation_spearman_rho_continuous_features":rho_concordance,
        "role":"SECONDARY_DESCRIPTIVE_ONLY",
    },

    "tree_dispersion":{
        "tree_count":TREE_COUNT,
        "tree_predict_calls":tree_predict_calls,
        "positive_association_target_count":sum(r["classification"]=="POSITIVE_ASSOCIATION" for r in dispersion_assoc_rows),
        "inconclusive_target_count":sum(r["classification"]=="INCONCLUSIVE" for r in dispersion_assoc_rows),
        "inverse_association_target_count":sum(r["classification"]=="INVERSE_ASSOCIATION" for r in dispersion_assoc_rows),
        "calibrated_interval_claim":"PROHIBITED",
    },

    "support_distance_error":{
        "spearman_rho":rho_support,
        "ci_low_2p5":lo_support,
        "ci_high_97p5":hi_support,
        "classification":support_class,
        "role":"VALIDITY_DIAGNOSTIC_ONLY",
    },

    "guardrails":{
        "surrogate_model_fit_calls":0,
        "surrogate_model_refit_calls":0,
        "generic_ice_status_permutation":"EXCLUDED",
        "new_counterfactual_icing_analysis":"NONE",
        "causal_feature_importance_claim":"PROHIBITED",
        "calibrated_predictive_interval_claim":"PROHIBITED",
        "max_abs_time_identity_difference":max_time_diff,
    },

    "model_sha256":sha256_file(model_path),
    "support_reference_sha256":sha256_file(support_npz_path),
}

with open(summary_out,"w",encoding="utf-8") as f:
    json.dump(summary,f,indent=2)

print("[P2-A7B] execution complete",flush=True)
print(json.dumps(summary,separators=(",",":")),flush=True)
