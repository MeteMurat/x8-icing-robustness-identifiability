import csv
import hashlib
import json
import math
import os
import platform
import sys

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

joined_path, a4d_contract_path, model_path, training_identity_path, manifest_path = sys.argv[1:6]

with open(a4d_contract_path, "r", encoding="utf-8-sig") as f:
    C = json.load(f)

R = C["final_refit_protocol"]
features = list(R["feature_preprocessing"]["exact_feature_vector"])
targets = list(R["target_preprocessing"]["exact_target_vector"])
est = dict(R["estimator"])

if R["model_family"] != "EXTRA_TREES":
    raise RuntimeError("Model-family drift")
if R["selected_config_id"] != "ET_FULL_LEAF1":
    raise RuntimeError("Selected-config drift")
if int(R["fit_count"]) != 1:
    raise RuntimeError("Fit-count contract drift")
if len(features) != 9 or len(targets) != 6:
    raise RuntimeError("Feature/target schema drift")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()

rows = []
with open(joined_path, "r", encoding="utf-8-sig", newline="") as f:
    r = csv.DictReader(f)
    required = ["configuration", "pair_group", "flight_id"] + features + targets
    missing = [c for c in required if c not in (r.fieldnames or [])]
    if missing:
        raise RuntimeError("Missing development fields: " + repr(missing))

    for rec in r:
        fid = int(rec["flight_id"])
        if fid >= 4:
            raise RuntimeError("LOCKED VALIDATION LEAKAGE IN DEVELOPMENT ARTIFACT")
        rows.append(rec)

if len(rows) != 23210:
    raise RuntimeError(f"Expected 23210 development rows; observed {len(rows)}")

X = np.asarray([[float(r[c]) for c in features] for r in rows], dtype=np.float64)
Y = np.asarray([[float(r[c]) for c in targets] for r in rows], dtype=np.float64)
groups = np.asarray([r["pair_group"] for r in rows], dtype=object)
configs = np.asarray([r["configuration"].strip().lower() for r in rows], dtype=object)
flight_ids = np.asarray([int(r["flight_id"]) for r in rows], dtype=int)

if sorted(set(flight_ids.tolist())) != [1, 2, 3]:
    raise RuntimeError("Unexpected development flight IDs")
if len(set(groups.tolist())) != 27:
    raise RuntimeError("Expected exactly 27 exact-pair groups")
if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
    raise RuntimeError("Non-finite development features/targets")

# Pair/configuration-balanced weights.
ug = sorted(set(groups.tolist()))
w = np.zeros(len(rows), dtype=np.float64)

for pair in ug:
    mg = (groups == pair)
    cfgset = set(configs[mg].tolist())
    if cfgset != {"clean", "iced"}:
        raise RuntimeError(f"Pair {pair} lacks exact clean/iced support: {cfgset}")

    for cfg in ("clean", "iced"):
        idx = np.where(mg & (configs == cfg))[0]
        if len(idx) == 0:
            raise RuntimeError("Empty pair/configuration cell")
        w[idx] = 0.5 / len(ug) / len(idx)

w *= len(w) / w.sum()

if np.any(w <= 0) or not np.all(np.isfinite(w)):
    raise RuntimeError("Invalid final training weights")

# Development-only weighted target standardization.
ws = w.sum()
y_mu = (w[:, None] * Y).sum(axis=0) / ws
y_var = (w[:, None] * (Y - y_mu) ** 2).sum(axis=0) / ws
y_sd = np.sqrt(y_var)

if not np.all(np.isfinite(y_mu)) or not np.all(np.isfinite(y_sd)) or np.any(y_sd <= 1e-12):
    raise RuntimeError("Invalid weighted target standardization")

Yz = (Y - y_mu) / y_sd

max_depth = est["max_depth"]
if max_depth is not None:
    max_depth = int(max_depth)

model = ExtraTreesRegressor(
    n_estimators=int(est["n_estimators"]),
    criterion=str(est["criterion"]),
    max_depth=max_depth,
    min_samples_leaf=int(est["min_samples_leaf"]),
    max_features=float(est["max_features"]),
    bootstrap=bool(est["bootstrap"]),
    random_state=int(est["random_state"]),
    n_jobs=int(est["n_jobs"]),
)

# EXACTLY ONE AUTHORIZED FINAL FIT.
fit_count = 0
model.fit(X, Yz, sample_weight=w)
fit_count += 1

if fit_count != 1:
    raise RuntimeError("Final model fit count is not exactly one")

# Structural checks only; no prediction and no performance metric computation.
if len(model.estimators_) != 256:
    raise RuntimeError("Final forest tree-count mismatch")
if int(model.n_features_in_) != 9:
    raise RuntimeError("Final model feature-count mismatch")
if int(model.n_outputs_) != 6:
    raise RuntimeError("Final model output-count mismatch")

bundle = {
    "model": model,
    "feature_names": features,
    "target_names": targets,
    "target_weighted_mean": y_mu,
    "target_weighted_sd": y_sd,
    "selected_config_id": R["selected_config_id"],
    "estimator_hyperparameters": model.get_params(deep=False),
    "sample_weighting": R["sample_weighting"],
    "training_data_sha256": sha256_file(joined_path),
    "parent_contract_sha256": C.get("_contract_sha256_external", None),
    "training_rows": len(rows),
    "exact_pair_groups": len(ug),
    "allowed_flight_ids": [1, 2, 3],
    "fit_count": fit_count,
    "prediction_count_at_freeze": 0,
    "performance_metric_count_at_freeze": 0,
}

joblib.dump(bundle, model_path, compress=3)

identity = {
    "development_rows": len(rows),
    "exact_pair_groups": len(ug),
    "flight_ids": sorted(set(flight_ids.tolist())),
    "configuration_values": sorted(set(configs.tolist())),
    "feature_count": len(features),
    "target_count": len(targets),
    "sample_weight_min": float(w.min()),
    "sample_weight_max": float(w.max()),
    "sample_weight_mean": float(w.mean()),
    "sample_weight_sum": float(w.sum()),
    "joined_sha256": sha256_file(joined_path),
    "locked_validation_numeric_rows_read": 0,
    "prediction_count": 0,
    "performance_metric_count": 0,
}
with open(training_identity_path, "w", encoding="utf-8") as f:
    json.dump(identity, f, indent=2)

manifest = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "numpy_version": np.__version__,
    "sklearn_version": sklearn.__version__,
    "joblib_version": joblib.__version__,
    "model_class": type(model).__name__,
    "selected_config_id": R["selected_config_id"],
    "tree_count": len(model.estimators_),
    "n_features_in": int(model.n_features_in_),
    "n_outputs": int(model.n_outputs_),
    "fit_count": fit_count,
    "prediction_count": 0,
    "performance_metric_count": 0,
    "model_bundle_sha256": sha256_file(model_path),
    "training_identity_sha256": sha256_file(training_identity_path),
    "locked_validation_numeric_rows_read": 0,
}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, separators=(",", ":")))
