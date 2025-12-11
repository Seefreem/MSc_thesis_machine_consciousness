import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train linear probes (LogisticRegression) with 10-fold CV."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./_datasets/HPC/",
        help="Base data directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name / subdirectory ",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default="imdb_sms_interval_1_pairs_with_activations.json",
        help="Target JSON file name",
    )
    parser.add_argument(
        "--label_feature",
        type=str,
        default="label_context_1",
        help="Name of the numerical label field to use (label_context_1 or label_context_2)",
    )
    parser.add_argument(
        "--layer_idx",
        type=str,
        default=None,
        help='Layer index (e.g. "0", "5") or "all" to train probes on all layers',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=45,
        help="Random seed",
    )
    parser.add_argument(
        "--C",
        type=str,
        default="1.0",
        help="Inverse of regularization strength for LogisticRegression (larger C = less regularization)",
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default='sen_w_t1',
        help='Candidate tasks: sen_w_t1, sen_w_t2, sen_w_b, lay_w_t1, lay_w_t2, lay_w_b, and selective_attention'
    ) 
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)


def load_metadata(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def infer_dims(first_hidden_path: str):
    hs = np.load(first_hidden_path, mmap_mode="r")
    n_tokens, n_layers, feat_dim = hs.shape
    return n_tokens, n_layers, feat_dim


def extract_features(
    data,
    base_dir: str,
    label_feature: str,
    layer_indices,
):
    """
    Extract features for the specified layers and labels from all samples.

    For each sample:
      - Load hidden_states_file (shape [all_tokens, n_layers, features])
      - Take token at index (len(all_tokens) - 10), or 0 if < 10
      - For each requested layer, take that layer's feature vector
    """
    n_samples = len(data)

    # Use the first file to determine n_layers and feat_dim
    first_hidden_path = data[0]["hidden_states_file"]
    _, n_layers_total, feat_dim = infer_dims(first_hidden_path)

    # Resolve which layers to use
    if layer_indices is None or len(layer_indices) == 0:
        raise ValueError("No layer indices provided.")

    for l in layer_indices:
        if not (0 <= l < n_layers_total):
            raise ValueError(f"Layer index {l} out of range [0, {n_layers_total - 1}]")

    # Allocate feature arrays per layer
    X_by_layer = {
        l: np.zeros((n_samples, feat_dim), dtype=np.float32) for l in layer_indices
    }

    # Labels
    y = np.zeros((n_samples,), dtype=np.float32)

    for i, sample in enumerate(data):
        hidden_path = sample["hidden_states_file"]
        # hidden_path = os.path.join(base_dir, hidden_file)

        hs = np.load(hidden_path, mmap_mode="r")  # [all_tokens, n_layers, features]
        n_tokens = hs.shape[0]
        token_idx = n_tokens - 10
        if token_idx < 0:
            token_idx = 0

        for l in layer_indices:
            X_by_layer[l][i, :] = hs[token_idx, l, :]

        y[i] = float(sample[label_feature])

    return X_by_layer, y


def train_probes_for_layer_logreg(
    C,
    X: np.ndarray,
    y: np.ndarray,
    layer_idx: int,
    out_dir: str,
    seed: int = 45,
    n_ite: int = 1000,
):
    """
    Train 10-fold CV logistic regression probes with per-fold StandardScaler.

    Returns:
      - train_losses: np.ndarray [n_folds]
      - val_losses:   np.ndarray [n_folds]
      - weight_vectors: list of np.ndarray of shape [D+1] (coef + intercept)
    """
    set_seed(seed)

    n_samples, input_dim = X.shape
    y = np.array(y, dtype=np.int32)  # assume labels are 0/1

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))

    n_folds = len(splits)
    train_losses = np.zeros((n_folds,), dtype=np.float32)
    val_losses = np.zeros((n_folds,), dtype=np.float32)
    weight_vectors = []
    acc_of_train = []
    acc_of_val = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"[Layer {layer_idx}] Fold {fold_idx + 1}/{n_folds}...")

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        # Per-fold normalization (fit on train only)
        scaler = StandardScaler()
        if C != np.inf:
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            print('Feature normalized!')
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
        
        # Classic logistic regression
        clf = LogisticRegression(
            penalty="l2",
            C=C, # C=np.inf results in unpenalized logistic regression.
            solver="lbfgs",
            max_iter=n_ite,
            random_state=seed,
        )
        clf.fit(X_train_scaled, y_train)
        # print(X_train_scaled.shape)
        # print(X_train_scaled[0, :])
        # print('mean', np.mean(X_train_scaled, axis=0))
        train_pred_label = clf.predict(X_train_scaled)
        val_pred_label   = clf.predict(X_val_scaled)
        acc_of_train.append(sum(train_pred_label == y_train)/len(y_train))
        acc_of_val.append(sum(val_pred_label == y_val)/len(y_val))

        print(
            f"  Fold {fold_idx + 1}: "
            # f"train_log_loss={train_ll:.4f}, val_log_loss={val_ll:.4f}"
            f"avg. acc. on train: {acc_of_train[-1]:.4f}; avg. acc. on val: {acc_of_val[-1]:.4f}"
        )

        # Save scaler + classifier for this fold
        model_bundle = {
            "scaler": scaler,
            "classifier": clf,
        }
        probe_path = os.path.join(
            out_dir, f"logreg_layer{layer_idx}_fold{fold_idx}.joblib"
        )
        joblib.dump(model_bundle, probe_path)

        # Weight vector for cosine similarity (coef + intercept)
        # Binary LR: coef_ shape [1, D], intercept_ shape [1]
        w = clf.coef_.reshape(-1)        # [D]
        b = clf.intercept_.reshape(-1)   # [1]
        wb = np.concatenate([w, b], axis=0)
        weight_vectors.append(wb)
    print('acc_of_train: ', acc_of_train)
    print('acc_of_val: ', acc_of_val)
    print(f'The standard deviation of the accuracy on the training sets is {np.array(acc_of_train).std():.4f}\n'
        f'The mean of the accuracy on the training sets is {np.array(acc_of_train).mean():.4f}'
        )
    print(f'The standard deviation of the accuracy on the validation sets is {np.array(acc_of_val).std():.4f}\n'
        f'The mean of the accuracy on the validation sets is {np.array(acc_of_val).mean():.4f}'
        )

    return acc_of_train, acc_of_val, weight_vectors

def plot_probe_similarity(weight_vectors, out_dir: str, layer_idx: int):
    """
    Compute cosine similarity between the 10 probes and plot a matrix
    (confusion-matrix-like visualization) with similarity values in each cell.
    """
    os.makedirs(out_dir, exist_ok=True)

    W = np.stack(weight_vectors, axis=0)  # [n_folds, D+1]
    sim = cosine_similarity(W)            # [n_folds, n_folds]

    plt.figure()
    im = plt.imshow(sim, vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Cosine similarity")

    n_folds = sim.shape[0]
    ticks = np.arange(n_folds)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    # Add similarity scores on each cell
    # Use a threshold so text color contrasts with the background
    thresh = (sim.max() + sim.min()) / 2.0
    for i in range(n_folds):
        for j in range(n_folds):
            val = sim[i, j]
            text_color = "white" if val > thresh else "black"
            plt.text(
                j,                # x position (column)
                i,                # y position (row)
                f"{val:.2f}",     # text
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    plt.xlabel("Fold")
    plt.ylabel("Fold")
    plt.title(f"Cosine similarity of probes (Layer {layer_idx})")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"probe_similarity_layer{layer_idx}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    np.save(os.path.join(out_dir, f"probe_similarity_layer{layer_idx}.npy"), sim)

def plot_acc_curves(acc_data, layer_indices, out_dir: str):
    """
    Plot mean ± std of train/val.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_mean = np.array([acc_data[idx][0] for idx in layer_indices ])
    train_std = np.array([acc_data[idx][1] for idx in layer_indices ])
    val_mean = np.array([acc_data[idx][2] for idx in layer_indices ])
    val_std = np.array([acc_data[idx][3] for idx in layer_indices ])

    plt.figure()
    plt.plot(layer_indices, train_mean, label="Train Acc.")
    plt.fill_between(
        layer_indices, train_mean - train_std, train_mean + train_std, alpha=0.2
    )

    plt.plot(layer_indices, val_mean, label="Validation Acc.")
    plt.fill_between(layer_indices, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.xlabel("Layer_index")
    plt.ylabel("Acc.")
    plt.title(f"The accuracy curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"acc_curves_layer.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # Also save raw loss arrays
    np.savez(
        os.path.join(out_dir, f"acc_mean_std.npz"),
        train_mean=train_mean,
        train_std=train_std,
        val_mean=val_mean,
        val_std=val_std,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    base_dir = os.path.join(args.data_dir, args.model_name, args.task)
    target_file = str(args.task) + "_" + args.target_file
    json_path = os.path.join(base_dir, target_file)

    print(f"Loading metadata from {json_path}")
    data = load_metadata(json_path)
    n_samples = len(data)
    print(f"Loaded {n_samples} samples.")

    # Determine layers to train
    first_hidden_path = data[0]["hidden_states_file"]
    _, n_layers_total, _ = infer_dims(first_hidden_path)

    if args.layer_idx is None:
        raise ValueError(
            'Please provide --layer_idx as an integer or "all" (default is None).'
        )

    if args.layer_idx.lower() == "all":
        layer_indices = list(range(n_layers_total))
    else:
        layer_indices = [int(args.layer_idx)]

    print(f"Will train probes for layers: {layer_indices}")

    # Extract features
    X_by_layer, y = extract_features(
        data=data,
        base_dir=base_dir,
        label_feature=args.label_feature,
        layer_indices=layer_indices,
    )
    if args.C == 'np.inf':
        C = np.inf
        n_ite=15
        probes_dir = os.path.join(base_dir, "linear_probes_logreg_inf")
    else:
        C = float(args.C)
        n_ite=10
        probes_dir = os.path.join(base_dir, "linear_probes_logreg_normalized")
    os.makedirs(probes_dir, exist_ok=True)
    layer_acc = {}
    for layer_idx in layer_indices:
        print("=" * 80)
        print(f"Training LogisticRegression probes for layer {layer_idx}")
        print("=" * 80)

        X = X_by_layer[layer_idx]
        layer_out_dir = os.path.join(probes_dir, f"layer_{layer_idx}")
        os.makedirs(layer_out_dir, exist_ok=True)
        print(X.shape)
        print(X[0])
        acc_of_train, acc_of_val, weight_vectors = train_probes_for_layer_logreg(
            X=X,
            y=y,
            layer_idx=layer_idx,
            C=C,
            out_dir=layer_out_dir,
            seed=args.seed,
            n_ite = n_ite
        )
        # plot_loss_summary(train_losses, val_losses, layer_out_dir, layer_idx)
        plot_probe_similarity(weight_vectors, layer_out_dir, layer_idx)
        layer_acc[layer_idx] = [np.mean(acc_of_train), np.std(acc_of_train), 
            np.mean(acc_of_val), np.std(acc_of_val)]
        print(f"Finished layer {layer_idx}. Outputs saved to {layer_out_dir}")
    # Plot acc curves
    plot_acc_curves(layer_acc, layer_indices, probes_dir)
    


if __name__ == "__main__":
    main()
