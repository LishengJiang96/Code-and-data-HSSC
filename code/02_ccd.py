# -*- coding: utf-8 -*-
"""Coupling coordination degree (CCD) calculation.

Reads results/subsystem_scores.xlsx and writes results/ccd.xlsx with C, T, D,
and the coordination grades.

Run:
    python code/02_ccd.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCORE_COLUMNS = {
    "CR": "Carbon reduction score",
    "PC": "Pollution control score",
    "GE": "Greening score",
    "GR": "Growth score",
}


def ccd_grade(d: float) -> str:
    grades = [
        (0.1, "Extremely disordered"), (0.2, "Severely disordered"),
        (0.3, "Moderately disordered"), (0.4, "Slightly disordered"),
        (0.5, "Fragile coordination"), (0.6, "Barely coordination"),
        (0.7, "Primary coordination"), (0.8, "Moderate coordination"),
        (0.9, "Good coordination"), (1.01, "Quality coordination"),
    ]
    for upper, label in grades:
        if d <= upper:
            return label
    return "Quality coordination"


def compute_ccd(scores: pd.DataFrame) -> pd.DataFrame:
    cols = [SCORE_COLUMNS[k] for k in ("CR", "PC", "GE", "GR")]
    U = scores[cols].astype(float).to_numpy()
    k = U.shape[1]
    pair = np.array([sum(abs(U[i, a] - U[i, b])
                         for a in range(k) for b in range(a + 1, k))
                     for i in range(len(U))])
    prod = np.prod(U / U.max(axis=1, keepdims=True), axis=1) ** (1.0 / (k - 1))
    C = np.sqrt((1.0 - pair / (k * (k - 1) / 2.0)) * prod)
    T = U.mean(axis=1)
    D = np.sqrt(C * T)
    out = scores.copy()
    out["Coupling degree C"] = C
    out["Coupling grade"] = [ccd_grade(v) for v in C]
    out["Comprehensive development index T"] = T
    out["Coupling coordination degree D"] = D
    out["Coordination grade"] = [ccd_grade(v) for v in D]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CCD calculation")
    parser.add_argument("--scores", default=os.path.join(ROOT, "results", "subsystem_scores.xlsx"))
    parser.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    scores = pd.read_excel(args.scores)
    ccd = compute_ccd(scores)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "ccd.xlsx")
    ccd.to_excel(path, index=False)
    print("Saved:", path)


if __name__ == "__main__":
    main()
