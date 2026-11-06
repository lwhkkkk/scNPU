"""Explain score discrepancies; diagnostic only, never alters validation gates."""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import linalg


def norm_report(x):
    return {"max_abs": float(np.max(np.abs(x))), "frobenius": float(np.linalg.norm(x))}


def diagnose(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    a = ad.read_h5ad(reference / "brain_1m_pca.h5ad", backed="r")
    b = ad.read_h5ad(candidate / "brain_1m_pca.h5ad", backed="r")
    la32 = np.asarray(a.varm["PCs"], dtype=np.float64)
    lb32 = np.asarray(b.varm["PCs"], dtype=np.float64)
    fa = pd.read_csv(reference / "pca_loadings.csv.gz", index_col=0, float_precision="round_trip")
    fb = pd.read_csv(candidate / "pca_loadings.csv.gz", index_col=0, float_precision="round_trip")
    assert fa.index.equals(fb.index)
    la, lb = fa.to_numpy(), fb.to_numpy()
    ma, mb = a.var.pca_mean_log1p.to_numpy(), b.var.pca_mean_log1p.to_numpy()
    da, db = a.var.pca_std_log1p.to_numpy(), b.var.pca_std_log1p.to_numpy()
    sa = np.load(reference / "pca_scores.npy", mmap_mode="r")
    sb = np.load(candidate / "pca_scores.npy", mmap_mode="r")
    x = np.load(reference / "log1p_hvg.npy", mmap_mode="r")
    r32, _ = linalg.orthogonal_procrustes(lb32, la32)
    r64, _ = linalg.orthogonal_procrustes(lb, la)
    linear, _, _, _ = linalg.lstsq(lb, la)
    qa, ra = linalg.qr(la, mode="economic")
    qb, rb = linalg.qr(lb, mode="economic")
    rq, _ = linalg.orthogonal_procrustes(qb, qa)
    maxima = {"h5ad_rotation": (0., 0, 0), "csv_rotation": (0., 0, 0), "linear_map": (0., 0, 0)}
    top = []
    for lo in range(0, len(sa), 8192):
        av = np.asarray(sa[lo:lo+8192], dtype=np.float64)
        bv = np.asarray(sb[lo:lo+8192], dtype=np.float64)
        for name, transform in (("h5ad_rotation", r32), ("csv_rotation", r64), ("linear_map", linear)):
            diff = bv @ transform - av
            row, pc = np.unravel_index(np.abs(diff).argmax(), diff.shape)
            value = float(abs(diff[row, pc]))
            if value > maxima[name][0]:
                maxima[name] = (value, int(lo + row), int(pc))
            if name == "h5ad_rotation":
                top.append((value, int(lo + row), int(pc)))
    top.sort(reverse=True)
    selected = sorted({v[1] for v in top[:20]} | {v[1] for v in maxima.values()})
    xv = np.asarray(x[selected], dtype=np.float64)
    av, bv = np.asarray(sa[selected], dtype=np.float64), np.asarray(sb[selected], dtype=np.float64)
    pa = (xv @ (la / da[:, None])) - ma @ (la / da[:, None])
    pb = (xv @ (lb / db[:, None])) - mb @ (lb / db[:, None])
    za = (xv - ma) / da
    terms = {
        "actual_aligned_difference": bv @ r32 - av,
        "combined_projection_roundoff": (bv - pb) @ r32 - (av - pa),
        "ideal_direction_and_statistics": pb @ r32 - pa,
        "rotation_using_rounded_loadings": bv @ (r32 - r64),
        "outside_subspace_loading_residual": za @ (lb @ linear - la),
        "nonorthogonal_loading_component": za @ (lb @ (r32 - linear)),
        "statistics_difference": (pb - za @ lb) @ r32,
        "baseline_projection_roundoff": av - pa,
        "candidate_projection_roundoff": bv - pb,
    }
    terms["decomposition_residual"] = terms["actual_aligned_difference"] - terms["combined_projection_roundoff"] - terms["outside_subspace_loading_residual"] - terms["nonorthogonal_loading_component"] - terms["statistics_difference"]
    # Canonical QR coordinates are an explanatory experiment, not a replacement
    # for the already declared artifact comparison metric.
    normalized_diff = (bv @ linalg.inv(rb)) @ rq - (av @ linalg.inv(ra))
    records = []
    for error, row, pc in top[:10]:
        i = selected.index(row)
        records.append({"row_zero_based": row, "pc_one_based": pc+1, "barcode": str(a.obs_names[row]),
                        "baseline_score": float(av[i, pc]), "candidate_aligned_score": float((bv[i] @ r32)[pc]),
                        "baseline_row_max_abs_score": float(np.max(np.abs(av[i]))),
                        "absolute_error": error, "relative_error_to_element": float(error/max(abs(av[i,pc]), 1e-30)),
                        "terms_at_element": {key: float(value[i, pc]) for key, value in terms.items()}})
    k = la.shape[1]
    report = {
        "reference": str(reference), "candidate": str(candidate),
        "diagnostic_only": True, "original_gate": {"aligned_score_max_abs_limit": .002, "changed": False},
        "full_score_scan": {key: {"max_abs": value[0], "row_zero_based": value[1], "pc_one_based": value[2]+1} for key, value in maxima.items()},
        "loadings": {"baseline_csv_vs_h5ad": norm_report(la-la32), "candidate_csv_vs_h5ad": norm_report(lb-lb32),
                     "baseline_orthogonality_error": norm_report(la.T @ la - np.eye(k)),
                     "candidate_orthogonality_error": norm_report(lb.T @ lb - np.eye(k)),
                     "aligned_residual": norm_report(lb @ r32 - la),
                     "linear_fit_residual": norm_report(lb @ linear - la),
                     "rotation_difference": norm_report(r32-r64)},
        "statistics": {"mean_difference": norm_report(ma-mb), "std_difference": norm_report(da-db), "minimum_std": float(min(da.min(),db.min()))},
        "sample_scope": "Up to 20 block-worst rows plus global maxima; not all-cell FP64 reprojection",
        "sample_count": len(selected), "sample_terms": {key: norm_report(value) for key,value in terms.items()},
        "sample_qr_coordinate_difference": norm_report(normalized_diff), "worst_rows": records,
    }
    a.file.close()
    b.file.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = diagnose(args.reference, args.candidate)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
