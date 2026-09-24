"""Executed historical tree configurations, without fitting."""

SEED = 20260909


def config(family, weight=1.0, alpha=None):
    if family == "CAT":
        return dict(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            thread_count=4,
            task_type="CPU",
            loss_function="Logloss",
            bootstrap_type="No",
            random_strength=0,
            l2_leaf_reg=3,
            border_count=254,
            random_seed=SEED,
            allow_writing_files=False,
            verbose=False,
            class_weights=[1.0, float(weight)],
        )
    if family == "XGB":
        return dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            n_jobs=4,
            tree_method="hist",
            device="cpu",
            objective="binary:logistic",
            min_child_weight=1,
            reg_lambda=1,
            reg_alpha=0,
            gamma=0,
            max_bin=256,
            random_state=SEED,
            scale_pos_weight=float(weight),
        )
    d = dict(
        n_estimators=300,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        n_jobs=4,
        subsample=1.0,
        colsample_bytree=1.0,
        subsample_freq=0,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=0.0,
        max_bin=255,
        random_state=SEED,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        device_type="cpu",
    )
    if alpha is None:
        d.update(objective="binary", scale_pos_weight=float(weight))
    else:
        d.update(objective="quantile", alpha=alpha)
    return d


def expert_columns(regime):
    return (
        list(range(308)) + list(range(320, 328))
        if regime == "R"
        else list(range(4, 17)) + list(range(149, 161)) + list(range(293, 328))
    )
