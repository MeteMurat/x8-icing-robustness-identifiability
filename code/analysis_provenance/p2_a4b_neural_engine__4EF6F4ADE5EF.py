import csv
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

joined_path, outer_plan_path, inner_plan_path, a4a_contract_path, a3_outer_path, a3_pair_path, inner_out, selection_out, outer_seed_out, outer_out, pair_target_out, pair_cmp_out, target_cmp_out, prediction_out, curve_out, summary_out, env_out = sys.argv[1:18]

with open(a4a_contract_path,"r",encoding="utf-8-sig") as f:
    C=json.load(f)

FEATURES=list(C["training_protocol"]["feature_preprocessing"]["columns"])
TARGETS=list(C["training_protocol"]["target_preprocessing"]["columns"])
GRID=list(C["architecture_grid"])
TP=C["training_protocol"]
SP=C["selection_protocol"]

EPOCHS=int(TP["epochs"])
BATCH=int(TP["batch_size"])
CLIP=float(TP["gradient_clip_global_norm"])
INNER_SEEDS=[int(x) for x in TP["inner_tuning_seeds"]]
OUTER_SEEDS=[int(x) for x in TP["outer_evaluation_seeds"]]
THREADS=int(TP["torch_num_threads"])
ETA_MIN=float(TP["learning_rate_schedule"]["eta_min"])
DEVICE=torch.device("cpu")

if len(GRID)!=6 or INNER_SEEDS!=[20260919,20260920] or OUTER_SEEDS!=[20260919,20260920,20260921]:
    raise RuntimeError("P2-A4A frozen grid/seed contract drift")
if EPOCHS!=120 or BATCH!=512 or THREADS!=2:
    raise RuntimeError("P2-A4A training protocol drift")

torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.use_deterministic_algorithms(True)

# --------------------------------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def write_csv(path, records):
    if not records:
        raise RuntimeError("Cannot write empty artifact: "+path)
    fields=[]
    seen=set()
    for r in records:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with open(path,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(records)

# --------------------------------------------------------------------------------------------------
# Load development-only data
# --------------------------------------------------------------------------------------------------
rows=[]
with open(joined_path,"r",encoding="utf-8-sig",newline="") as f:
    r=csv.DictReader(f)
    need=["configuration","file_name","sample_index","TIME","pair_group","flight_id"]+FEATURES+TARGETS
    missing=[c for c in need if c not in (r.fieldnames or [])]
    if missing:
        raise RuntimeError("Development join missing fields: "+repr(missing))
    for rec in r:
        if int(rec["flight_id"])>=4:
            raise RuntimeError("LOCKED VALIDATION LEAKAGE IN DEVELOPMENT ARTIFACT")
        rows.append(rec)

X=np.asarray([[float(r[c]) for c in FEATURES] for r in rows],dtype=np.float64)
Y=np.asarray([[float(r[c]) for c in TARGETS] for r in rows],dtype=np.float64)
groups=np.asarray([r["pair_group"] for r in rows],dtype=object)
configs=np.asarray([r["configuration"].strip().lower() for r in rows],dtype=object)
flight_ids=np.asarray([int(r["flight_id"]) for r in rows],dtype=int)

if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
    raise RuntimeError("Non-finite development features/targets")
if sorted(set(flight_ids.tolist()))!=[1,2,3]:
    raise RuntimeError("Unexpected development IDs")
if len(set(groups.tolist()))!=27:
    raise RuntimeError("Expected 27 exact-pair groups")

# --------------------------------------------------------------------------------------------------
# Frozen splits
# --------------------------------------------------------------------------------------------------
with open(outer_plan_path,"r",encoding="utf-8-sig",newline="") as f:
    OUTER_PLAN=list(csv.DictReader(f))
with open(inner_plan_path,"r",encoding="utf-8-sig",newline="") as f:
    INNER_PLAN=list(csv.DictReader(f))

def outer_groups(hold,role):
    fold=f"OUTER_HOLD_ID_{hold}"
    return {
        r["pair_group"] for r in OUTER_PLAN
        if r["outer_fold"]==fold and r["role"]==role
    }

def inner_groups(hold,fold,role):
    return {
        r["pair_group"] for r in INNER_PLAN
        if int(r["outer_hold_id"])==hold
        and int(r["inner_fold"])==fold
        and r["role"]==role
    }

def idx_for(gs):
    return np.where(np.isin(groups,list(gs)))[0]

# --------------------------------------------------------------------------------------------------
# Pair/config-balanced weights and weighted scaling
# --------------------------------------------------------------------------------------------------
def training_weights(index):
    index=np.asarray(index,dtype=int)
    g=groups[index]
    c=configs[index]
    ug=sorted(set(g.tolist()))
    w=np.zeros(len(index),dtype=np.float64)

    for pair in ug:
        mg=(g==pair)
        cfgset=set(c[mg].tolist())
        if cfgset!={"clean","iced"}:
            raise RuntimeError(f"Training pair {pair} lacks clean/iced support: {cfgset}")
        for cfg in ("clean","iced"):
            loc=np.where(mg & (c==cfg))[0]
            if len(loc)==0:
                raise RuntimeError("Empty pair/config training cell")
            w[loc]=0.5/len(ug)/len(loc)

    w *= len(w)/w.sum()
    if np.any(w<=0) or not np.all(np.isfinite(w)):
        raise RuntimeError("Invalid training weights")
    return w

def weighted_mean_sd(a,w):
    ws=w.sum()
    mu=(w[:,None]*a).sum(axis=0)/ws
    var=(w[:,None]*(a-mu)**2).sum(axis=0)/ws
    sd=np.sqrt(var)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)) or np.any(sd<=1e-12):
        raise RuntimeError("Invalid weighted scale")
    return mu,sd

# --------------------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        layers=[]
        d=len(FEATURES)
        for h in hidden:
            layers.append(nn.Linear(d,int(h)))
            layers.append(nn.SiLU())
            d=int(h)
        layers.append(nn.Linear(d,len(TARGETS)))
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)

def parameter_count(hidden):
    d=len(FEATURES)
    total=0
    for h in hidden:
        total += d*int(h)+int(h)
        d=int(h)
    total += d*len(TARGETS)+len(TARGETS)
    return total

# --------------------------------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------------------------------
def pair_target_metrics(y_true,y_pred,eval_groups,train_sd):
    rec=[]
    for pair in sorted(set(eval_groups.tolist())):
        m=(eval_groups==pair)
        err=y_pred[m]-y_true[m]
        rmse=np.sqrt(np.mean(err*err,axis=0))
        mae=np.mean(np.abs(err),axis=0)
        nrmse=rmse/train_sd
        for j,t in enumerate(TARGETS):
            rec.append({
                "pair_group":pair,
                "target":t,
                "rmse":float(rmse[j]),
                "mae":float(mae[j]),
                "train_sd":float(train_sd[j]),
                "nrmse":float(nrmse[j]),
            })
    score=float(np.mean([r["nrmse"] for r in rec]))
    return score,rec

# --------------------------------------------------------------------------------------------------
# One deterministic fit
# --------------------------------------------------------------------------------------------------
def train_predict(cfg,train_idx,pred_idx,seed):
    set_seed(seed)

    tr=np.asarray(train_idx,dtype=int)
    pr=np.asarray(pred_idx,dtype=int)
    w=training_weights(tr)

    xmu,xsd=weighted_mean_sd(X[tr],w)
    ymu,ysd=weighted_mean_sd(Y[tr],w)

    Xtr=((X[tr]-xmu)/xsd).astype(np.float32)
    Ytr=((Y[tr]-ymu)/ysd).astype(np.float32)
    Xpr=((X[pr]-xmu)/xsd).astype(np.float32)
    Wtr=w.astype(np.float32)

    ds=TensorDataset(
        torch.from_numpy(Xtr),
        torch.from_numpy(Ytr),
        torch.from_numpy(Wtr),
    )

    gen=torch.Generator()
    gen.manual_seed(seed)

    loader=DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        generator=gen,
        num_workers=0,
        drop_last=False,
    )

    model=MLP(cfg["hidden_layers"]).to(DEVICE)

    opt=torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        betas=(0.9,0.999),
        eps=1e-8,
        weight_decay=float(cfg["weight_decay"]),
    )

    sched=torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=EPOCHS,
        eta_min=ETA_MIN,
    )

    first_loss=None
    last_loss=None

    model.train()
    for epoch in range(EPOCHS):
        num=0.0
        den=0.0
        for xb,yb,wb in loader:
            xb=xb.to(DEVICE)
            yb=yb.to(DEVICE)
            wb=wb.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            pred=model(xb)
            per_sample=torch.mean((pred-yb)**2,dim=1)
            loss=torch.sum(wb*per_sample)/torch.sum(wb)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite neural loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
            opt.step()

            num += float(torch.sum(wb*per_sample).detach().cpu())
            den += float(torch.sum(wb).detach().cpu())

        epoch_loss=num/den
        if first_loss is None:
            first_loss=epoch_loss
        last_loss=epoch_loss
        sched.step()

    model.eval()
    with torch.no_grad():
        zp=model(torch.from_numpy(Xpr).to(DEVICE)).cpu().numpy().astype(np.float64)

    yp=zp*ysd+ymu
    if not np.all(np.isfinite(yp)):
        raise RuntimeError("Non-finite neural predictions")

    return yp,ysd,float(first_loss),float(last_loss)

# --------------------------------------------------------------------------------------------------
# Load frozen ExtraTrees comparator
# --------------------------------------------------------------------------------------------------
with open(a3_outer_path,"r",encoding="utf-8-sig",newline="") as f:
    a3_outer=[r for r in csv.DictReader(f) if r["family"]=="EXTRA_TREES"]

with open(a3_pair_path,"r",encoding="utf-8-sig",newline="") as f:
    a3_pair=[r for r in csv.DictReader(f) if r["family"]=="EXTRA_TREES"]

if len(a3_outer)!=3:
    raise RuntimeError("Expected 3 frozen ExtraTrees outer rows")
if len(a3_pair)!=27*6:
    raise RuntimeError(f"Expected 162 frozen ExtraTrees pair-target rows, observed {len(a3_pair)}")

et_outer={int(r["outer_hold_id"]):float(r["outer_primary_score"]) for r in a3_outer}
et_pair={(r["pair_group"],r["target"]):float(r["nrmse"]) for r in a3_pair}

# --------------------------------------------------------------------------------------------------
# Execute frozen nested neural experiment
# --------------------------------------------------------------------------------------------------
inner_records=[]
selection_records=[]
outer_seed_records=[]
outer_records=[]
pair_target_records=[]
prediction_records=[]
curve_records=[]

train_run_count=0
t0=time.time()

for hold in (1,2,3):
    outer_train=outer_groups(hold,"TRAINING")
    outer_val=outer_groups(hold,"VALIDATION")
    if outer_train & outer_val:
        raise RuntimeError("Outer group leakage")

    config_scores=[]

    for cfg in GRID:
        config_id=cfg["config_id"]
        run_scores=[]

        for inner_fold in (1,2,3):
            ig_train=inner_groups(hold,inner_fold,"INNER_TRAINING")
            ig_val=inner_groups(hold,inner_fold,"INNER_VALIDATION")

            if ig_train & ig_val:
                raise RuntimeError("Inner group leakage")
            if ig_train | ig_val != outer_train:
                raise RuntimeError("Inner split does not exactly cover outer training groups")

            tr_idx=idx_for(ig_train)
            va_idx=idx_for(ig_val)

            for seed in INNER_SEEDS:
                yp,train_sd,loss_first,loss_last=train_predict(cfg,tr_idx,va_idx,seed)
                train_run_count += 1

                score,_=pair_target_metrics(Y[va_idx],yp,groups[va_idx],train_sd)
                run_scores.append(score)

                inner_records.append({
                    "outer_hold_id":hold,
                    "config_id":config_id,
                    "inner_fold":inner_fold,
                    "seed":seed,
                    "primary_score":score,
                    "train_groups":len(ig_train),
                    "validation_groups":len(ig_val),
                    "train_rows":len(tr_idx),
                    "validation_rows":len(va_idx),
                    "loss_epoch1":loss_first,
                    "loss_epoch120":loss_last,
                    "parameter_count":parameter_count(cfg["hidden_layers"]),
                })

                curve_records.append({
                    "phase":"INNER",
                    "outer_hold_id":hold,
                    "inner_fold":inner_fold,
                    "config_id":config_id,
                    "seed":seed,
                    "loss_epoch1":loss_first,
                    "loss_epoch120":loss_last,
                    "loss_ratio_final_to_initial":loss_last/loss_first if loss_first>0 else float("nan"),
                })

        config_scores.append({
            "config_id":config_id,
            "score":float(np.mean(run_scores)),
            "parameter_count":parameter_count(cfg["hidden_layers"]),
            "cfg":cfg,
        })

    config_scores.sort(key=lambda r:(r["score"],r["parameter_count"],r["config_id"]))
    selected=config_scores[0]
    selected_cfg=selected["cfg"]

    selection_records.append({
        "outer_hold_id":hold,
        "selected_config_id":selected["config_id"],
        "selected_inner_mean_score":selected["score"],
        "parameter_count":selected["parameter_count"],
        "runner_up_config_id":config_scores[1]["config_id"],
        "runner_up_inner_mean_score":config_scores[1]["score"],
        "selection_margin":config_scores[1]["score"]-selected["score"],
    })

    tr_idx=idx_for(outer_train)
    va_idx=idx_for(outer_val)

    seed_predictions=[]
    seed_train_sd=None

    for seed in OUTER_SEEDS:
        yp,train_sd,loss_first,loss_last=train_predict(selected_cfg,tr_idx,va_idx,seed)
        train_run_count += 1
        seed_predictions.append(yp)

        if seed_train_sd is None:
            seed_train_sd=train_sd
        elif not np.allclose(seed_train_sd,train_sd,rtol=0,atol=1e-12):
            raise RuntimeError("Training scale drift across outer seeds")

        seed_score,_=pair_target_metrics(Y[va_idx],yp,groups[va_idx],train_sd)

        outer_seed_records.append({
            "outer_hold_id":hold,
            "selected_config_id":selected["config_id"],
            "seed":seed,
            "primary_score":seed_score,
            "loss_epoch1":loss_first,
            "loss_epoch120":loss_last,
            "train_groups":len(outer_train),
            "validation_groups":len(outer_val),
            "train_rows":len(tr_idx),
            "validation_rows":len(va_idx),
        })

        curve_records.append({
            "phase":"OUTER",
            "outer_hold_id":hold,
            "inner_fold":"",
            "config_id":selected["config_id"],
            "seed":seed,
            "loss_epoch1":loss_first,
            "loss_epoch120":loss_last,
            "loss_ratio_final_to_initial":loss_last/loss_first if loss_first>0 else float("nan"),
        })

    stack=np.stack(seed_predictions,axis=0)
    agg=np.mean(stack,axis=0)
    pred_sd=np.std(stack,axis=0,ddof=1)

    outer_score,pt=pair_target_metrics(Y[va_idx],agg,groups[va_idx],seed_train_sd)
    seed_scores=[r["primary_score"] for r in outer_seed_records if r["outer_hold_id"]==hold]

    outer_records.append({
        "outer_hold_id":hold,
        "selected_config_id":selected["config_id"],
        "selected_inner_mean_score":selected["score"],
        "neural_outer_primary_score":outer_score,
        "seed_primary_score_mean":float(np.mean(seed_scores)),
        "seed_primary_score_sd":float(np.std(seed_scores,ddof=1)),
        "extra_trees_outer_primary_score":et_outer[hold],
        "relative_improvement_vs_extra_trees":(et_outer[hold]-outer_score)/et_outer[hold],
        "neural_beats_extra_trees":outer_score < et_outer[hold],
    })

    for r in pt:
        key=(r["pair_group"],r["target"])
        if key not in et_pair:
            raise RuntimeError("Missing frozen ExtraTrees pair-target comparator")
        et=et_pair[key]
        pair_target_records.append({
            "outer_hold_id":hold,
            "selected_config_id":selected["config_id"],
            **r,
            "extra_trees_nrmse":et,
            "relative_improvement_vs_extra_trees":(et-r["nrmse"])/et,
            "neural_beats_extra_trees":r["nrmse"] < et,
        })

    # Save one outer prediction per development row: each pair belongs to exactly one outer holdout.
    for local_i,global_i in enumerate(va_idx):
        base={
            "outer_hold_id":hold,
            "selected_config_id":selected["config_id"],
            "configuration":rows[global_i]["configuration"],
            "file_name":rows[global_i]["file_name"],
            "sample_index":rows[global_i]["sample_index"],
            "TIME":rows[global_i]["TIME"],
            "pair_group":rows[global_i]["pair_group"],
            "flight_id":rows[global_i]["flight_id"],
        }
        for j,t in enumerate(TARGETS):
            base["true_"+t]=float(Y[global_i,j])
            base["pred_"+t]=float(agg[local_i,j])
            base["seed_sd_"+t]=float(pred_sd[local_i,j])
        prediction_records.append(base)

if train_run_count != 117:
    raise RuntimeError(f"Frozen neural training-run accounting mismatch: expected 117, got {train_run_count}")

# --------------------------------------------------------------------------------------------------
# Comparison against frozen ExtraTrees
# --------------------------------------------------------------------------------------------------
neural_outer_mean=float(np.mean([r["neural_outer_primary_score"] for r in outer_records]))
et_outer_mean=float(np.mean([r["extra_trees_outer_primary_score"] for r in outer_records]))
outer_improved=sum(1 for r in outer_records if r["neural_beats_extra_trees"])

# Pair macro across six targets
pair_acc=defaultdict(list)
et_pair_acc=defaultdict(list)
for r in pair_target_records:
    pair_acc[r["pair_group"]].append(float(r["nrmse"]))
    et_pair_acc[r["pair_group"]].append(float(r["extra_trees_nrmse"]))

pair_cmp=[]
for pg in sorted(pair_acc):
    n=float(np.mean(pair_acc[pg]))
    e=float(np.mean(et_pair_acc[pg]))
    pair_cmp.append({
        "pair_group":pg,
        "neural_pair_macro_nrmse":n,
        "extra_trees_pair_macro_nrmse":e,
        "relative_improvement_vs_extra_trees":(e-n)/e,
        "neural_beats_extra_trees":n<e,
    })

pair_improved=sum(1 for r in pair_cmp if r["neural_beats_extra_trees"])

# Target macro across 27 pairs
target_cmp=[]
for t in TARGETS:
    rr=[r for r in pair_target_records if r["target"]==t]
    n=float(np.mean([float(r["nrmse"]) for r in rr]))
    e=float(np.mean([float(r["extra_trees_nrmse"]) for r in rr]))
    target_cmp.append({
        "target":t,
        "neural_pair_macro_nrmse":n,
        "extra_trees_pair_macro_nrmse":e,
        "relative_improvement_vs_extra_trees":(e-n)/e,
        "neural_beats_extra_trees":n<e,
    })

target_improved=sum(1 for r in target_cmp if r["neural_beats_extra_trees"])

mean_better=neural_outer_mean < et_outer_mean
broad=(mean_better and outer_improved==3 and pair_improved>=14 and target_improved>=4)

if broad:
    classification="NEURAL_BROADLY_SUPERIOR_TO_CLASSICAL"
elif mean_better:
    classification="NEURAL_MEAN_ADVANTAGE_HETEROGENEOUS"
else:
    classification="NEURAL_NOT_BETTER_THAN_CLASSICAL"

# --------------------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------------------
write_csv(inner_out,inner_records)
write_csv(selection_out,selection_records)
write_csv(outer_seed_out,outer_seed_records)
write_csv(outer_out,outer_records)
write_csv(pair_target_out,pair_target_records)
write_csv(pair_cmp_out,pair_cmp)
write_csv(target_cmp_out,target_cmp)
write_csv(prediction_out,prediction_records)
write_csv(curve_out,curve_records)

env={
    "python":sys.version,
    "platform":platform.platform(),
    "torch":torch.__version__,
    "numpy":np.__version__,
    "sklearn":sklearn.__version__,
    "device":"cpu",
    "torch_num_threads":torch.get_num_threads(),
}
open(env_out,"w",encoding="utf-8").write(json.dumps(env,indent=2))

summary={
    "development_rows":len(rows),
    "exact_pair_groups":len(set(groups.tolist())),
    "neural_configurations":len(GRID),
    "neural_training_runs":train_run_count,
    "inner_result_rows":len(inner_records),
    "outer_seed_result_rows":len(outer_seed_records),
    "outer_aggregate_result_rows":len(outer_records),
    "prediction_rows":len(prediction_records),
    "selected_configs":selection_records,
    "neural_outer_mean_primary_score":neural_outer_mean,
    "extra_trees_outer_mean_primary_score":et_outer_mean,
    "relative_improvement_vs_extra_trees":(et_outer_mean-neural_outer_mean)/et_outer_mean,
    "outer_folds_improved_vs_extra_trees":outer_improved,
    "outer_folds_total":3,
    "pair_groups_improved_vs_extra_trees":pair_improved,
    "pair_groups_total":27,
    "targets_improved_vs_extra_trees":target_improved,
    "targets_total":6,
    "development_classification":classification,
    "locked_validation_numeric_rows_read":0,
    "source_mutation":"NONE",
    "elapsed_seconds":time.time()-t0,
}
open(summary_out,"w",encoding="utf-8").write(json.dumps(summary,indent=2))
print(json.dumps(summary,separators=(",",":")))
