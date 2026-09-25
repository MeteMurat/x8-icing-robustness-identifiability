import json
import sys
import joblib

bundle = joblib.load(sys.argv[1])
model = bundle["model"]

result = {
    "bundle_keys_ok": all(k in bundle for k in [
        "model","feature_names","target_names","target_weighted_mean","target_weighted_sd",
        "selected_config_id","estimator_hyperparameters","training_data_sha256",
        "training_rows","exact_pair_groups","fit_count"
    ]),
    "model_class": type(model).__name__,
    "selected_config_id": bundle["selected_config_id"],
    "feature_count": len(bundle["feature_names"]),
    "target_count": len(bundle["target_names"]),
    "tree_count": len(model.estimators_),
    "fit_count": int(bundle["fit_count"]),
    "prediction_count_at_freeze": int(bundle["prediction_count_at_freeze"]),
    "performance_metric_count_at_freeze": int(bundle["performance_metric_count_at_freeze"]),
}
print(json.dumps(result, separators=(",", ":")))
