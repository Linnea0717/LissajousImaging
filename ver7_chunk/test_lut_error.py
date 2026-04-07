
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def exact_mapping(phase, direction, max_idx, shift=0.0):
    """
    和原本 getMappingIdxWeight 完全相同的邏輯，用 numpy 實作。
    phase: 0.0 ~ 1.0
    """
    theta = np.pi * phase
    if direction == -1:
        theta = np.pi - theta

    x_phys = np.cos(theta)
    weight = np.sqrt(np.maximum(0.0, 1.0 - x_phys ** 2))

    x_norm = (x_phys + 1.0) * 0.5
    x_pix  = x_norm * max_idx

    if direction == +1:
        x_pix -= shift
    else:
        x_pix += shift

    idx = np.clip(np.round(x_pix).astype(np.int32), 0, max_idx - 1)
    return idx, weight


def build_lut(size):
    theta   = np.linspace(0.0, np.pi, size, dtype=np.float64)
    lut_cos = np.cos(theta).astype(np.float32)
    lut_sin = np.sqrt(np.maximum(0.0, 1.0 - lut_cos.astype(np.float64) ** 2)).astype(np.float32)
    return lut_cos, lut_sin


def lut_mapping(phase, direction, max_idx, lut_cos, lut_sin, shift=0.0):
    lut_size = len(lut_cos)

    p = phase.copy()
    if direction == -1:
        p = 1.0 - p

    lut_idx = np.clip(np.round(p * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)

    x_phys = lut_cos[lut_idx].astype(np.float64)
    weight = lut_sin[lut_idx].astype(np.float64)

    x_norm = (x_phys + 1.0) * 0.5
    x_pix  = x_norm * max_idx

    if direction == +1:
        x_pix -= shift
    else:
        x_pix += shift

    idx = np.clip(np.round(x_pix).astype(np.int32), 0, max_idx - 1)
    return idx, weight


def compute_error(lut_size, W_out, n_samples=100_000, direction=+1, shift=0.0):
    rng   = np.random.default_rng(42)
    phase = rng.uniform(0.0, 1.0, n_samples)

    theta_exact = np.pi * phase
    if direction == -1:
        theta_exact = np.pi - theta_exact
    x_phys_exact  = np.cos(theta_exact)
    weight_exact  = np.sqrt(np.maximum(0.0, 1.0 - x_phys_exact ** 2))
    x_pix_exact   = (x_phys_exact + 1.0) * 0.5 * W_out
    if direction == +1:
        x_pix_exact -= shift
    else:
        x_pix_exact += shift
    idx_exact = np.clip(np.round(x_pix_exact).astype(np.int32), 0, W_out - 1)

    lut_cos, lut_sin = build_lut(lut_size)
    idx_lut, weight_lut = lut_mapping(phase, direction, W_out, lut_cos, lut_sin, shift)

    idx_diff    = np.abs(idx_lut - idx_exact)
    weight_diff = np.abs(weight_lut - weight_exact.astype(np.float32))

    return {
        'idx_mae':         float(idx_diff.mean()),
        'idx_max':         int(idx_diff.max()),
        'idx_diff_pct':    float((idx_diff > 0).mean() * 100),
        'weight_mae':      float(weight_diff.mean()),
        'weight_max':      float(weight_diff.max()),
    }


LUT_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
W_OUTS    = [512, 1024, 2048]

if __name__ == "__main__":

    print(f"{'LUT':>6}  {'W_out':>6}  "
          f"{'idx MAE':>9}  {'idx MAX':>8}  "
          f"{'!=exact':>9}  {'>1px':>8}  "
          f"{'w MAE':>10}  {'w MAX':>10}")
    print("-" * 85)

    results = {}
    for lut_size, W_out in product(LUT_SIZES, W_OUTS):
        err = compute_error(lut_size, W_out)
        results[(lut_size, W_out)] = err
        print(f"{lut_size:>6}  {W_out:>6}  "
              f"{err['idx_mae']:>9.4f}  {err['idx_max']:>8d}  "
              f"{err['idx_diff_pct']:>8.2f}%  "
              f"{err['weight_mae']:>10.6f}  {err['weight_max']:>10.6f}")


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("LUT Mapping Error vs Exact cos/sqrt", fontsize=13)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for ax, W_out, color in zip(axes, W_OUTS, colors):
        maes = [results[(s, W_out)]['idx_mae']      for s in LUT_SIZES]
        maxs = [results[(s, W_out)]['idx_max']       for s in LUT_SIZES]
        pcts = [results[(s, W_out)]['idx_diff_pct']  for s in LUT_SIZES]

        ax2 = ax.twinx()
        ax.plot(LUT_SIZES, maes, 'o-', color=color,      label='MAE (px)',      linewidth=2)
        ax.plot(LUT_SIZES, maxs, 's--', color=color,     label='Max err (px)',  linewidth=1.5, alpha=0.6)
        ax2.plot(LUT_SIZES, pcts, '^:', color='gray',    label='≠ exact (%)',   linewidth=1.5)

        ax.axhline(y=0.5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='0.5 px')
        ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5)

        ax.set_xscale('log', base=2)
        ax.set_xlabel("LUT size")
        ax.set_ylabel("Index error (pixels)", color=color)
        ax2.set_ylabel("% samples ≠ exact", color='gray')
        ax.set_title(f"W_out = {W_out}")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

        ax.set_xticks(LUT_SIZES)
        ax.set_xticklabels([str(s) for s in LUT_SIZES], rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lut_error.png", dpi=150, bbox_inches='tight')
    print("\n[INFO] Saved plot to lut_error.png")
    plt.show()