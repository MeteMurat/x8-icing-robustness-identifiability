import csv
import hashlib
import json
import math
import sys

import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors

joined_path, json_out, npz_out = sys.argv[1:4]

CONT = ["Va_EKF_CG","alpha","beta","p","q","r","aileron","elevator"]
ICE = "ice_status"

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest().upper()

def fnum(x,name):
    v=float(x)
    if not math.isfinite(v):
        raise RuntimeError(f"Non-finite {name}")
    return v

def weighted_quantile(values, weights, q):
    order=np.argsort(values)
    v=np.asarray(values,dtype=float)[order]
    w=np.asarray(weights,dtype=float)[order]
    cw=np.cumsum(w)
    cutoff=q*cw[-1]
    idx=int(np.searchsorted(cw,cutoff,side="left"))
    idx=min(max(idx,0),len(v)-1)
    return float(v[idx])

rows=[]
with open(joined_path,"r",encoding="utf-8-sig",newline="") as f:
    r=csv.DictReader(f)
    required=["configuration","pair_group","flight_id",ICE]+CONT
    missing=[c for c in required if c not in (r.fieldnames or [])]
    if missing:
        raise RuntimeError("Development join missing fields: "+repr(missing))
    rows=list(r)

if len(rows)!=23210:
    raise RuntimeError(f"Expected 23210 development rows, observed {len(rows)}")

cfg=np.asarray([r["configuration"].strip().lower() for r in rows],dtype=object)
grp=np.asarray([r["pair_group"] for r in rows],dtype=object)
fid=np.asarray([int(r["flight_id"]) for r in rows],dtype=int)
X=np.asarray([[fnum(r[c],c) for c in CONT] for r in rows],dtype=np.float64)
ice=np.asarray([fnum(r[ICE],ICE) for r in rows],dtype=np.float64)

if sorted(set(fid.tolist()))!=[1,2,3]:
    raise RuntimeError("Unexpected development IDs")
if len(set(grp.tolist()))!=27:
    raise RuntimeError("Expected 27 development exact-pair groups")
if set(cfg.tolist())!={"clean","iced"}:
    raise RuntimeError("Expected exactly clean/iced configurations")
if not np.all(np.isfinite(X)):
    raise RuntimeError("Non-finite continuous development features")

clean_ice=sorted(set(ice[cfg=="clean"].tolist()))
iced_ice=sorted(set(ice[cfg=="iced"].tolist()))
if len(clean_ice)!=1 or len(iced_ice)!=1:
    raise RuntimeError("ice_status must be constant within clean/iced development configurations")
if clean_ice[0]==iced_ice[0]:
    raise RuntimeError("clean/iced ice_status values are not distinct")

# Pair/config-balanced weights, identical concept to prior P2 training protocol.
ug=sorted(set(grp.tolist()))
w=np.zeros(len(rows),dtype=np.float64)
for pair in ug:
    mg=(grp==pair)
    for c in ("clean","iced"):
        idx=np.where(mg & (cfg==c))[0]
        if len(idx)==0:
            raise RuntimeError("Development exact pair lacks clean/iced support: "+pair)
        w[idx]=0.5/len(ug)/len(idx)

w *= len(w)/w.sum()
mu=(w[:,None]*X).sum(axis=0)/w.sum()
var=(w[:,None]*(X-mu)**2).sum(axis=0)/w.sum()
sd=np.sqrt(var)
if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)) or np.any(sd<=1e-12):
    raise RuntimeError("Invalid development support scaling")

Z=(X-mu)/sd
zc=Z[cfg=="clean"]
zi=Z[cfg=="iced"]
wc=w[cfg=="clean"]
wi=w[cfg=="iced"]

K=10
if len(zc)<K or len(zi)<K:
    raise RuntimeError("Insufficient development rows for kNN support calibration")

nn_clean=NearestNeighbors(n_neighbors=K,algorithm="auto",n_jobs=2).fit(zc)
nn_iced=NearestNeighbors(n_neighbors=K,algorithm="auto",n_jobs=2).fit(zi)

# Cross-configuration support-distance reference.
d_clean_to_iced=np.mean(nn_iced.kneighbors(zc,return_distance=True)[0],axis=1)
d_iced_to_clean=np.mean(nn_clean.kneighbors(zi,return_distance=True)[0],axis=1)

cross=np.concatenate([d_clean_to_iced,d_iced_to_clean])
cross_w=np.concatenate([wc,wi])

q90=weighted_quantile(cross,cross_w,0.90)
q95=weighted_quantile(cross,cross_w,0.95)
q99=weighted_quantile(cross,cross_w,0.99)

result={
    "development_rows":len(rows),
    "exact_pair_groups":len(ug),
    "continuous_features":CONT,
    "continuous_feature_weighted_mean":mu.tolist(),
    "continuous_feature_weighted_sd":sd.tolist(),
    "clean_ice_status_value":float(clean_ice[0]),
    "iced_ice_status_value":float(iced_ice[0]),
    "knn_k":K,
    "distance_metric":"euclidean_on_development_weighted_standardized_continuous_features",
    "cross_configuration_distance_statistic":"mean_distance_to_10_nearest_opposite_configuration_development_rows",
    "threshold_weighting":"pair_and_configuration_balanced",
    "threshold_q90":q90,
    "threshold_q95":q95,
    "threshold_q99":q99,
    "primary_common_support_rule":"d_clean<=q95 AND d_iced<=q95",
    "sensitivity_common_support_rules":[
        "d_clean<=q90 AND d_iced<=q90",
        "d_clean<=q99 AND d_iced<=q99"
    ],
    "development_join_sha256":sha256_file(joined_path),
    "sklearn_version":sklearn.__version__,
    "model_predictions_computed":0,
    "locked_validation_rows_opened":0
}

with open(json_out,"w",encoding="utf-8") as f:
    json.dump(result,f,indent=2)

np.savez_compressed(
    npz_out,
    continuous_features=np.asarray(CONT,dtype="U32"),
    weighted_mean=mu,
    weighted_sd=sd,
    clean_reference_scaled=zc,
    iced_reference_scaled=zi,
    clean_ice_status_value=np.asarray([clean_ice[0]],dtype=float),
    iced_ice_status_value=np.asarray([iced_ice[0]],dtype=float),
    k=np.asarray([K],dtype=int),
    q90=np.asarray([q90],dtype=float),
    q95=np.asarray([q95],dtype=float),
    q99=np.asarray([q99],dtype=float),
)

print(json.dumps(result,separators=(",",":")))
