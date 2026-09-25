import csv
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor

joined_path, outer_path, nested_path, preanalysis_path, inner_out, outer_out, pair_out, family_out, env_out, summary_out = sys.argv[1:11]

with open(preanalysis_path,"r",encoding="utf-8-sig") as f:
    pre = json.load(f)

FEATURES = list(pre["features"])
TARGETS = list(pre["targets"])
SEED = int(pre["random_seed"])

# --------------------------------------------------------------------------------------
# Load frozen development rows only.
# --------------------------------------------------------------------------------------
rows=[]
with open(joined_path,"r",encoding="utf-8-sig",newline="") as f:
    r=csv.DictReader(f)
    required=["configuration","file_name","pair_group","flight_id"]+FEATURES+TARGETS
    missing=[c for c in required if c not in (r.fieldnames or [])]
    if missing:
        raise RuntimeError("Joined development artifact missing: "+repr(missing))

    for rec in r:
        fid=int(rec["flight_id"])
        if fid>=4:
            raise RuntimeError("LOCKED VALIDATION LEAKAGE in joined development data")
        rows.append(rec)

if not rows:
    raise RuntimeError("No development rows")

X=np.asarray([[float(r[c]) for c in FEATURES] for r in rows],dtype=float)
Y=np.asarray([[float(r[c]) for c in TARGETS] for r in rows],dtype=float)
groups=np.asarray([r["pair_group"] for r in rows],dtype=object)
configs=np.asarray([r["configuration"].strip().lower() for r in rows],dtype=object)
flight_ids=np.asarray([int(r["flight_id"]) for r in rows],dtype=int)

if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
    raise RuntimeError("Non-finite development X/Y")

unique_groups=sorted(set(groups.tolist()))
if len(unique_groups)!=27:
    raise RuntimeError(f"Expected 27 groups, got {len(unique_groups)}")
if sorted(set(flight_ids.tolist()))!=[1,2,3]:
    raise RuntimeError("Unexpected development flight IDs")

# --------------------------------------------------------------------------------------
# Load frozen fold plans.
# --------------------------------------------------------------------------------------
outer_assign=[]
with open(outer_path,"r",encoding="utf-8-sig",newline="") as f:
    outer_assign=list(csv.DictReader(f))

nested_assign=[]
with open(nested_path,"r",encoding="utf-8-sig",newline="") as f:
    nested_assign=list(csv.DictReader(f))

# --------------------------------------------------------------------------------------
# Pair/config-balanced sample weights, rescaled to mean one.
# --------------------------------------------------------------------------------------
def training_weights(index):
    index=np.asarray(index,dtype=int)
    g=groups[index]
    c=configs[index]
    ug=sorted(set(g.tolist()))
    if not ug:
        raise RuntimeError("Empty training partition")

    w=np.zeros(len(index),dtype=float)

    for pair in ug:
        mask_g=(g==pair)
        cfgs=sorted(set(c[mask_g].tolist()))
        if set(cfgs)!={"clean","iced"}:
            raise RuntimeError(f"Training group {pair} lacks clean/iced support: {cfgs}")

        for cfg in ("clean","iced"):
            loc=np.where(mask_g & (c==cfg))[0]
            if len(loc)==0:
                raise RuntimeError("Empty pair/config cell")
            w[loc]=0.5/len(ug)/len(loc)

    # Preserve relative balance but normalize mean weight to one so Ridge alpha has conventional scale.
    w *= len(w)/w.sum()

    if not np.all(np.isfinite(w)) or np.any(w<=0):
        raise RuntimeError("Invalid training weights")

    return w

def weighted_mean_sd(y,w):
    w=np.asarray(w,dtype=float)
    y=np.asarray(y,dtype=float)
    ws=w.sum()
    mu=(w[:,None]*y).sum(axis=0)/ws
    var=(w[:,None]*(y-mu)**2).sum(axis=0)/ws
    sd=np.sqrt(var)

    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)):
        raise RuntimeError("Non-finite weighted target scale")
    if np.any(sd<=1e-12):
        raise RuntimeError("Training target weighted SD <=1e-12")

    return mu,sd

# --------------------------------------------------------------------------------------
# Metric: pair macro x target macro train-scale NRMSE.
# --------------------------------------------------------------------------------------
def pair_target_metrics(y_true,y_pred,eval_groups,train_sd):
    y_true=np.asarray(y_true,dtype=float)
    y_pred=np.asarray(y_pred,dtype=float)
    out=[]
    for pair in sorted(set(eval_groups.tolist())):
        m=(eval_groups==pair)
        if not np.any(m):
            continue
        e=y_pred[m]-y_true[m]
        rmse=np.sqrt(np.mean(e*e,axis=0))
        mae=np.mean(np.abs(e),axis=0)
        nrmse=rmse/train_sd

        for j,t in enumerate(TARGETS):
            out.append({
                "pair_group":pair,
                "target":t,
                "rmse":float(rmse[j]),
                "mae":float(mae[j]),
                "train_sd":float(train_sd[j]),
                "nrmse":float(nrmse[j]),
            })

    if not out:
        raise RuntimeError("No pair metrics")

    score=float(np.mean([r["nrmse"] for r in out]))
    return score,out

# --------------------------------------------------------------------------------------
# Model fit/predict with frozen preprocessing.
# --------------------------------------------------------------------------------------
def fit_predict(family,cfg,train_idx,pred_idx):
    train_idx=np.asarray(train_idx,dtype=int)
    pred_idx=np.asarray(pred_idx,dtype=int)

    Xtr=X[train_idx]
    Xp=X[pred_idx]
    Ytr=Y[train_idx]
    w=training_weights(train_idx)

    ymu,ysd=weighted_mean_sd(Ytr,w)
    Ztr=(Ytr-ymu)/ysd

    if family=="OLS":
        xs=StandardScaler()
        xs.fit(Xtr,sample_weight=w)
        Xt=xs.transform(Xtr)
        Xv=xs.transform(Xp)

        model=LinearRegression()
        model.fit(Xt,Ztr,sample_weight=w)
        Zp=model.predict(Xv)

    elif family=="RIDGE":
        xs=StandardScaler()
        xs.fit(Xtr,sample_weight=w)
        Xt=xs.transform(Xtr)
        Xv=xs.transform(Xp)

        model=Ridge(alpha=float(cfg["alpha"]))
        model.fit(Xt,Ztr,sample_weight=w)
        Zp=model.predict(Xv)

    elif family=="POLY2_RIDGE":
        poly=PolynomialFeatures(degree=2,include_bias=False)
        Xt0=poly.fit_transform(Xtr)
        Xv0=poly.transform(Xp)

        xs=StandardScaler()
        xs.fit(Xt0,sample_weight=w)
        Xt=xs.transform(Xt0)
        Xv=xs.transform(Xv0)

        model=Ridge(alpha=float(cfg["alpha"]))
        model.fit(Xt,Ztr,sample_weight=w)
        Zp=model.predict(Xv)

    elif family=="EXTRA_TREES":
        model=ExtraTreesRegressor(
            n_estimators=256,
            random_state=SEED,
            n_jobs=2,
            criterion="squared_error",
            max_depth=cfg["max_depth"],
            min_samples_leaf=int(cfg["min_samples_leaf"]),
            max_features=float(cfg["max_features"]),
        )
        model.fit(Xtr,Ztr,sample_weight=w)
        Zp=model.predict(Xp)

    else:
        raise RuntimeError("Unknown family: "+family)

    Yp=Zp*ysd+ymu
    if not np.all(np.isfinite(Yp)):
        raise RuntimeError("Non-finite prediction")

    return Yp,ysd

# --------------------------------------------------------------------------------------
# Frozen grids from preanalysis.
# --------------------------------------------------------------------------------------
grids={}
for fam in pre["model_families"]:
    family=fam["family"]
    grids[family]=list(fam["grid"])

family_order=["OLS","RIDGE","POLY2_RIDGE","EXTRA_TREES"]

# --------------------------------------------------------------------------------------
# Helper to get group sets from frozen plans.
# --------------------------------------------------------------------------------------
def groups_for_outer(hold_id,role):
    fold=f"OUTER_HOLD_ID_{hold_id}"
    return {
        r["pair_group"] for r in outer_assign
        if r["outer_fold"]==fold and r["role"]==role
    }

def groups_for_inner(hold_id,inner_fold,role):
    return {
        r["pair_group"] for r in nested_assign
        if int(r["outer_hold_id"])==hold_id
        and int(r["inner_fold"])==inner_fold
        and r["role"]==role
    }

def indices_for_group_set(gs):
    return np.where(np.isin(groups,list(gs)))[0]

inner_records=[]
outer_records=[]
pair_records=[]
fit_count=0
t0=time.time()

for family in family_order:
    for hold_id in (1,2,3):
        outer_train_groups=groups_for_outer(hold_id,"TRAINING")
        outer_val_groups=groups_for_outer(hold_id,"VALIDATION")

        if outer_train_groups & outer_val_groups:
            raise RuntimeError("Outer group leakage")

        candidate_scores=[]

        for cfg in grids[family]:
            cfg_id=cfg["config_id"]
            inner_scores=[]

            for inner_fold in (1,2,3):
                tr_groups=groups_for_inner(hold_id,inner_fold,"INNER_TRAINING")
                va_groups=groups_for_inner(hold_id,inner_fold,"INNER_VALIDATION")

                if tr_groups & va_groups:
                    raise RuntimeError("Inner group leakage")
                if tr_groups | va_groups != outer_train_groups:
                    raise RuntimeError("Inner groups do not exactly cover outer training groups")

                tr_idx=indices_for_group_set(tr_groups)
                va_idx=indices_for_group_set(va_groups)

                yp,train_sd=fit_predict(family,cfg,tr_idx,va_idx)
                fit_count+=1

                score,_=pair_target_metrics(
                    Y[va_idx],
                    yp,
                    groups[va_idx],
                    train_sd
                )

                inner_scores.append(score)

                inner_records.append({
                    "family":family,
                    "outer_hold_id":hold_id,
                    "config_id":cfg_id,
                    "inner_fold":inner_fold,
                    "inner_score":score,
                    "train_groups":len(tr_groups),
                    "validation_groups":len(va_groups),
                    "train_rows":len(tr_idx),
                    "validation_rows":len(va_idx),
                })

            mean_score=float(np.mean(inner_scores))
            candidate_scores.append((mean_score,cfg_id,cfg))

        # deterministic tie-break by config id
        candidate_scores.sort(key=lambda x:(x[0],x[1]))
        selected_inner_score,selected_cfg_id,selected_cfg=candidate_scores[0]

        tr_idx=indices_for_group_set(outer_train_groups)
        va_idx=indices_for_group_set(outer_val_groups)

        yp,train_sd=fit_predict(family,selected_cfg,tr_idx,va_idx)
        fit_count+=1

        outer_score,pair_metrics=pair_target_metrics(
            Y[va_idx],
            yp,
            groups[va_idx],
            train_sd
        )

        # descriptive physical-unit per-target metrics over whole outer holdout
        residual=yp-Y[va_idx]
        target_rmse=np.sqrt(np.mean(residual**2,axis=0))
        target_mae=np.mean(np.abs(residual),axis=0)

        outer_records.append({
            "family":family,
            "outer_hold_id":hold_id,
            "selected_config_id":selected_cfg_id,
            "selected_inner_score":selected_inner_score,
            "outer_primary_score":outer_score,
            "train_groups":len(outer_train_groups),
            "validation_groups":len(outer_val_groups),
            "train_rows":len(tr_idx),
            "validation_rows":len(va_idx),
            **{f"rmse_{TARGETS[j]}":float(target_rmse[j]) for j in range(len(TARGETS))},
            **{f"mae_{TARGETS[j]}":float(target_mae[j]) for j in range(len(TARGETS))},
        })

        for rec in pair_metrics:
            pair_records.append({
                "family":family,
                "outer_hold_id":hold_id,
                "selected_config_id":selected_cfg_id,
                **rec
            })

# --------------------------------------------------------------------------------------
# Family summaries.
# --------------------------------------------------------------------------------------
family_summary=[]
for family in family_order:
    rr=[r for r in outer_records if r["family"]==family]
    if len(rr)!=3:
        raise RuntimeError(f"Expected 3 outer results for {family}")

    scores=np.asarray([r["outer_primary_score"] for r in rr],dtype=float)

    family_summary.append({
        "family":family,
        "outer_mean_primary_score":float(np.mean(scores)),
        "outer_sd_primary_score":float(np.std(scores,ddof=1)),
        "outer_min_primary_score":float(np.min(scores)),
        "outer_max_primary_score":float(np.max(scores)),
        "outer_fold_count":len(scores),
        "selected_configs":"|".join(
            f"ID{r['outer_hold_id']}:{r['selected_config_id']}" for r in sorted(rr,key=lambda x:x["outer_hold_id"])
        )
    })

# Add descriptive improvement vs RIDGE mean.
ridge_mean=next(r["outer_mean_primary_score"] for r in family_summary if r["family"]=="RIDGE")
for r in family_summary:
    r["relative_improvement_vs_ridge"] = (
        float((ridge_mean-r["outer_mean_primary_score"])/ridge_mean)
        if ridge_mean!=0 else float("nan")
    )

family_summary.sort(key=lambda x:(x["outer_mean_primary_score"],x["family"]))
for rank,r in enumerate(family_summary,1):
    r["development_rank"]=rank

# --------------------------------------------------------------------------------------
# Write outputs.
# --------------------------------------------------------------------------------------
def write_csv(path,records):
    if not records:
        raise RuntimeError("Cannot write empty records: "+path)
    fields=[]
    seen=set()
    for r in records:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(records)

write_csv(inner_out,inner_records)
write_csv(outer_out,outer_records)
write_csv(pair_out,pair_records)
write_csv(family_out,family_summary)

env={
    "python":sys.version,
    "platform":platform.platform(),
    "numpy":np.__version__,
    "sklearn":sklearn.__version__,
    "random_seed":SEED,
    "extratrees_n_jobs":2,
}
open(env_out,"w",encoding="utf-8").write(json.dumps(env,indent=2))

summary={
    "development_rows":len(rows),
    "exact_pair_groups":len(unique_groups),
    "flight_ids":sorted(set(flight_ids.tolist())),
    "model_families":family_order,
    "total_model_fits":fit_count,
    "outer_result_rows":len(outer_records),
    "inner_result_rows":len(inner_records),
    "pair_target_error_rows":len(pair_records),
    "family_summary":family_summary,
    "best_development_family":family_summary[0]["family"],
    "best_development_score":family_summary[0]["outer_mean_primary_score"],
    "ridge_reference_score":ridge_mean,
    "locked_validation_numeric_rows_read":0,
    "neural_network_runs":0,
    "elapsed_seconds":time.time()-t0,
}
open(summary_out,"w",encoding="utf-8").write(json.dumps(summary,indent=2))
print(json.dumps(summary,separators=(",",":")))
