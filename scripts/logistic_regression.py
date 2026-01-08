from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

from spatialinfo import spatial_information as si


# -----------------------------
# helper: ANN model (corridor only)
# -----------------------------
def build_keras_model(input_size, n_corridors):
    inputs = tf.keras.Input(shape=(input_size,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    corridor_output = layers.Dense(
        n_corridors, activation="softmax", name="corridor_output"
    )(x)

    model = models.Model(inputs=inputs, outputs=corridor_output)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------------
# helper: logistic regression CV curve over C (train only)
# -----------------------------
def l1_logreg_cv_curve(X_train, y_train, C_grid, n_splits=5, random_state=42):
    """
    Returns:
        mean_acc: (len(C_grid),) mean CV accuracy on training set
        se_acc:   (len(C_grid),) standard error across folds
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",  # good for binary + L1
                    random_state=random_state,
                    max_iter=5000,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    mean_acc = np.zeros(len(C_grid), dtype=float)
    se_acc = np.zeros(len(C_grid), dtype=float)

    for i, C in enumerate(C_grid):
        pipe.set_params(clf__C=C)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        mean_acc[i] = scores.mean()
        se_acc[i] = scores.std(ddof=1) / np.sqrt(len(scores))

    return mean_acc, se_acc


# -----------------------------
# helper: fit logreg at a given C and evaluate on test
# -----------------------------
def fit_l1_logreg_and_test(X_train, y_train, X_test, y_test, C, random_state=42):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    random_state=random_state,
                    max_iter=5000,
                    C=C,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return accuracy_score(y_test, y_pred), pipe


def fit_l1_pipe(C, random_state=42):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    random_state=random_state,
                    max_iter=5000,
                    C=C,
                ),
            ),
        ]
    )


def frac_nonzero_at_C(X_train, y_train, C, eps=1e-8, random_state=42):
    """Fit on full training set at given C and return fraction of non-zero coefficients."""
    pipe = fit_l1_pipe(C, random_state=random_state)
    pipe.fit(X_train, y_train)

    coef = pipe.named_steps["clf"].coef_.ravel()
    return float(np.mean(np.abs(coef) > eps))


def frac_nonzero_curve(X_train, y_train, C_grid, eps=1e-8, random_state=42):
    """Return frac_nonzero for each C in C_grid."""
    return np.array(
        [
            frac_nonzero_at_C(X_train, y_train, C, eps=eps, random_state=random_state)
            for C in C_grid
        ]
    )


def pick_C_best_and_1se(C_grid, cv_mean, cv_se):
    """
    Returns:
        best_C: C with max mean CV
        C_1se: smallest-C (strongest reg) within 1 SE of best mean
    """
    best_idx = int(np.argmax(cv_mean))
    best_mean = float(cv_mean[best_idx])
    best_se = float(cv_se[best_idx])

    # 1-SE rule: choose simplest model whose mean is within 1 SE of the best
    threshold = best_mean - best_se

    # "simplest" for L1 means stronger regularization => smaller C
    eligible = np.where(cv_mean >= threshold)[0]
    idx_1se = int(eligible[0])  # because C_grid is increasing; earliest is smallest C

    return float(C_grid[best_idx]), float(C_grid[idx_1se]), best_idx, idx_1se


def selection_frequency_per_feature(
    X_train, y_train, C, n_splits=5, eps=1e-8, random_state=42
):
    """
    Selection frequency per feature across CV folds at fixed C.
    Output: freq array of shape (n_features,) with values in [0,1].
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_features = X_train.shape[1]
    counts = np.zeros(n_features, dtype=int)

    for tr_idx, _ in cv.split(X_train, y_train):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        pipe = fit_l1_pipe(C, random_state=random_state)
        pipe.fit(X_tr, y_tr)

        coef = pipe.named_steps["clf"].coef_.ravel()
        selected = np.abs(coef) > eps
        counts += selected.astype(int)

    return counts / n_splits


# -----------------------------
# main
# -----------------------------
data_path = Path("./data/processed_datasets")
datasets = [
    "20240903_Dc_1",
    "20240916_Dc_1",
    "20240930_Dc_1",
    "20241203_Dc",
    "20250722_Dc_2",
    "20250828_1300_DcLeft",
    "20250828_1300_DcRight",
    "20251030_Dc_1",
    "20251030_Dc_2",
]

# Common C grid for comparability across datasets
C_grid = np.logspace(-4, 2, 25)

# storage
results_rows = []
curves = (
    {}
)  # curves[dataset] = dict(C_grid=..., cv_mean=..., cv_se=..., logreg_test_by_C=...)

for dataset in datasets:
    # ---- load + preprocess exactly as you already do ----
    dff, bh = si.load_data(data_path / dataset)
    print("data loaded:", dataset)

    if len(bh.columns) < 6:
        bh = si.remove_interpolated_values(bh, n_corr=2)
        bh = si.add_trial_column(bh)
    else:
        bh.columns = [
            "timestamp",
            "trial",
            "trial index",
            "X",
            "Y",
            "velocity",
            "tail tracking framecount",
        ]

    results_df = si.trials_over_time(dff, bh, n_bins=50)
    X, (y_bins, y_corridors) = si.format_for_ann(results_df)

    # ---- split (note: you stratify by bins; OK if that’s your experimental choice) ----
    X_train, X_test, y_bins_train, y_bins_test, y_corridors_train, y_corridors_test = (
        train_test_split(X, y_bins, y_corridors, test_size=0.1, random_state=42)
    )

    # =========================================
    # 1) ANN (with early stopping to be principled)
    # =========================================
    keras_model = build_keras_model(input_size=X.shape[1], n_corridors=2)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
        )
    ]

    keras_model.fit(
        X_train,
        y_corridors_train,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred_corridors_proba = keras_model.predict(X_test, verbose=0)
    y_pred_corridors = np.argmax(y_pred_corridors_proba, axis=1)
    ann_test_acc = accuracy_score(y_corridors_test, y_pred_corridors)

    # =========================================
    # 2) Logistic regression (L1) – CV curve on train, choose best C, test once
    # =========================================
    cv_mean, cv_se = l1_logreg_cv_curve(
        X_train, y_corridors_train, C_grid=C_grid, n_splits=5, random_state=42
    )
    best_idx = int(np.argmax(cv_mean))
    best_C = float(C_grid[best_idx])

    logreg_test_acc, logreg_pipe = fit_l1_logreg_and_test(
        X_train, y_corridors_train, X_test, y_corridors_test, C=best_C, random_state=42
    )

    # Optional: also compute a "test curve" (descriptive only; not for tuning)
    # If you want to overlay curves, this can be useful — just don’t pick C from it.
    logreg_test_by_C = []
    for C in C_grid:
        acc, _ = fit_l1_logreg_and_test(
            X_train, y_corridors_train, X_test, y_corridors_test, C=C, random_state=42
        )
        logreg_test_by_C.append(acc)
    logreg_test_by_C = np.array(logreg_test_by_C)

    # --- after cv_mean, cv_se are computed ---
    best_C, C_1se, best_idx, idx_1se = pick_C_best_and_1se(C_grid, cv_mean, cv_se)

    # sparsity along the C grid (fit-on-full-train for each C)
    frac_nz = frac_nonzero_curve(
        X_train, y_corridors_train, C_grid, eps=1e-8, random_state=42
    )

    # selection frequency across folds at best_C (or at C_1se if you prefer more parsimony)
    sel_freq_best = selection_frequency_per_feature(
        X_train, y_corridors_train, C=best_C, n_splits=5, eps=1e-8, random_state=42
    )

    # store curve
    curves[dataset] = {
        "C_grid": C_grid,
        "cv_mean": cv_mean,
        "cv_se": cv_se,
        "best_C": best_C,
        "logreg_test_by_C": logreg_test_by_C,
        "best_C": best_C,
        "C_1se": C_1se,
        "best_idx": best_idx,
        "idx_1se": idx_1se,
        "frac_nonzero": frac_nz,
        "sel_freq_bestC": sel_freq_best,
    }

    # store summary row
    results_rows.append(
        {
            "dataset": dataset,
            "ann_test_acc": ann_test_acc,
            "logreg_l1_bestC": best_C,
            "logreg_l1_cv_best_mean_acc": float(cv_mean[best_idx]),
            "logreg_l1_test_acc_at_bestC": logreg_test_acc,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(X.shape[1]),
            "logreg_l1_C_1se": C_1se,
            "logreg_l1_frac_nonzero_at_bestC": float(frac_nz[best_idx]),
            "logreg_l1_frac_nonzero_at_1se": float(frac_nz[idx_1se]),
        }
    )

    print(f"{dataset}: ANN test acc={ann_test_acc:.4f}")
    print(f"L1 logreg test acc (CV-tuned)={logreg_test_acc:.4f} @ C={best_C:g}")


# =========================================
# Results table
# =========================================
results_df = pd.DataFrame(results_rows).sort_values("dataset")
print("\n=== Summary ===")
print(
    results_df[
        ["dataset", "ann_test_acc", "logreg_l1_test_acc_at_bestC", "logreg_l1_bestC"]
    ]
)

# -----------------------------
# Presentation-ready plotting defaults
# -----------------------------
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.titlesize": 20,
        "lines.linewidth": 2.5,
        "lines.markersize": 7,
    }
)

# -----------------------------
# Rename datasets for plots: Dataset 1, Dataset 2, ...
# -----------------------------
# Preserve your original order in `datasets`
pretty_names = {ds: f"Dataset {i+1}" for i, ds in enumerate(datasets)}

# Add pretty name column to results_df
results_df = results_df.copy()
results_df["dataset_pretty"] = results_df["dataset"].map(pretty_names)

# Output folder for figures
fig_dir = Path("./figures")
fig_dir.mkdir(parents=True, exist_ok=True)


# =========================================
# Figure 1: Overlay CV accuracy curves (train-CV) for logreg across datasets
# =========================================
fig1 = plt.figure(figsize=(11, 7))

for ds in datasets:
    Cg = curves[ds]["C_grid"]
    mu = curves[ds]["cv_mean"]
    se = curves[ds]["cv_se"]
    label = pretty_names[ds]

    plt.plot(np.log10(Cg), mu, label=label)
    plt.fill_between(np.log10(Cg), mu - se, mu + se, alpha=0.15)

plt.xlabel("log10(C)")
plt.ylabel("CV accuracy (train only)")
plt.title("L1 Logistic Regression: CV accuracy vs C")
plt.axhline(
    y=0.5, linestyle="--", linewidth=2, color="gray", alpha=0.7, label="Chance level"
)
plt.ylim(0.0, 1.02)
plt.legend(ncol=2, frameon=True)
plt.tight_layout()

fig1_path = fig_dir / "fig1_logreg_cv_accuracy_vs_C.svg"
fig1.savefig(fig1_path, format="svg")


# =========================================
# Figure 2: ANN vs LogReg (CV-tuned) test accuracy per dataset
# =========================================

# Keep the plotting order aligned with `datasets`
results_df_plot = results_df.set_index("dataset").loc[datasets].reset_index()

fig2 = plt.figure(figsize=(7, 7))

ann_acc = results_df_plot["ann_test_acc"].to_numpy()
logreg_acc = results_df_plot["logreg_l1_test_acc_at_bestC"].to_numpy()

# Boxplots
plt.boxplot(
    [logreg_acc, ann_acc],
    positions=[1, 2],
    widths=0.5,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor="lightgray", alpha=0.6),
    medianprops=dict(color="black", linewidth=2),
)

# Scatter individual datapoints
x_logreg = np.ones_like(logreg_acc) * 1
x_ann = np.ones_like(ann_acc) * 2

plt.scatter(x_logreg, logreg_acc, color="black", zorder=3)
plt.scatter(x_ann, ann_acc, color="black", zorder=3)

# Connect paired datapoints
for i in range(len(ann_acc)):
    plt.plot(
        [1, 2],
        [logreg_acc[i], ann_acc[i]],
        color="gray",
        alpha=0.6,
        linewidth=1.5,
        zorder=2,
    )

plt.xticks([1, 2], ["L1 Logistic Regression", "ANN"])
plt.ylabel("Test accuracy")
plt.title("ANN vs L1 Logistic Regression — Paired Comparison")
plt.ylim(0, 1.02)

plt.tight_layout()

fig2_path = fig_dir / "fig2_ann_vs_logreg_paired_boxplot.svg"
fig2.savefig(fig2_path, format="svg")


# =========================================
# Figure 3: Test-accuracy curves vs C for logreg (descriptive)
# =========================================
fig3 = plt.figure(figsize=(11, 7))

for ds in datasets:
    Cg = curves[ds]["C_grid"]
    test_curve = curves[ds]["logreg_test_by_C"]
    label = pretty_names[ds]
    plt.plot(np.log10(Cg), test_curve, label=label)

plt.xlabel("log10(C)")
plt.ylabel("Test accuracy (descriptive; not for tuning)")
plt.title("L1 Logistic Regression: Test accuracy vs C")
plt.axhline(
    y=0.5, linestyle="--", linewidth=2, color="gray", alpha=0.7, label="Chance level"
)
plt.ylim(0.0, 1.02)
plt.legend(ncol=2, frameon=True)
plt.tight_layout()

fig3_path = fig_dir / "fig3_logreg_test_accuracy_vs_C.svg"
fig3.savefig(fig3_path, format="svg")


# =========================================
# Figure 4: CV accuracy vs fraction non-zero (with best & 1-SE points)
# =========================================
fig4 = plt.figure(figsize=(11, 7))

for ds in datasets:
    fnz = curves[ds]["frac_nonzero"]
    mu = curves[ds]["cv_mean"]
    se = curves[ds]["cv_se"]

    label = pretty_names[ds]

    # main curve
    plt.plot(fnz, mu, label=label)
    plt.fill_between(fnz, mu - se, mu + se, alpha=0.15)

    # mark best and 1-SE points
    b = curves[ds]["best_idx"]
    s = curves[ds]["idx_1se"]
    plt.scatter([fnz[b]], [mu[b]], marker="o", zorder=5)
    plt.scatter([fnz[s]], [mu[s]], marker="s", zorder=5)

plt.xlabel("Fraction of non-zero coefficients")
plt.ylabel("CV accuracy (train only)")
plt.title(
    "L1 Logistic Regression: CV accuracy vs sparsity\n(circle = CV-optimal, square = 1-SE)"
)
plt.ylim(0, 1.02)
plt.xlim(0, 1.0)
plt.legend(ncol=2, frameon=True)
plt.tight_layout()

fig4_path = fig_dir / "fig4_cv_accuracy_vs_fraction_nonzero.svg"
fig4.savefig(fig4_path, format="svg")


print("Saved figures:")
print(" ", fig1_path)
print(" ", fig2_path)
print(" ", fig3_path)
print(" ", fig4_path)


summary_df = pd.DataFrame(results_rows)
summary_df.to_csv("results/logreg_dataset_summary.csv", index=False)

curve_rows = []

for i, C in enumerate(C_grid):
    curve_rows.append(
        {
            "dataset": dataset,
            "dataset_pretty": pretty_names[dataset],
            "C": C,
            "log10C": np.log10(C),
            "cv_mean_accuracy": cv_mean[i],
            "cv_se_accuracy": cv_se[i],
            "frac_nonzero": frac_nz[i],
            "test_accuracy": curves[dataset]["logreg_test_by_C"][i],
        }
    )

curve_df = pd.DataFrame(curve_rows)
curve_df.to_parquet("results/logreg_curves.parquet")

freq = curves[dataset]["sel_freq_bestC"]

feature_df = pd.DataFrame(
    {
        "dataset": dataset,
        "dataset_pretty": pretty_names[dataset],
        "feature_index": np.arange(len(freq)),
        "selection_frequency": freq,
        "selected_bestC": freq > 0,
    }
)

feature_df.to_parquet("results/logreg_selection_frequency.parquet")
