# -*- coding: utf-8 -*-
"""Subsystem calculation: data preparation, weights, and DNMA scores.

Reads data/panel_529.xlsx (sheets Normalized and MinMax), recovers the raw
growth-rate values, computes the entropy/CRITIC/uninorm weights and the
per-year DNMA subsystem scores, and writes the main-analysis outputs to results/.

Run:
    python code/01_subsystem.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Configuration (Table 1 of the manuscript)
# ---------------------------------------------------------------------------
INDICATORS_RELATIVE = [
    "Per capita CO2 emissions",
    "CO2 emission intensity",
    "Energy consumption intensity",
    "Energy consumption structure",
    "Industrial smoke and dust emission intensity",
    "Industrial SO2 emission intensity",
    "Industrial wastewater emission intensity",
    "Industrial solid waste treatment rate",
    "Fertilizer application intensity",
    "Environmental regulation intensity",
    "Environmental governance intensity",
    "Green invention patent density",
    "Green utility model patent density",
    "Per capita afforestation area",
    "Forest coverage rate",
    "Green coverage rate of built-up areas",
    "Per capita park green space area",
    "Per capita built-up area",
    "Per capita GDP growth rate",
    "Per capita disposable income growth rate",
    "Financial development level",
    "Social consumption level",
    "External openness level",
    "Number of public buses per 10,000 people",
    "Urbanization level",
    "Unemployment rate",
]
INDICATORS_ABSOLUTE = list(INDICATORS_RELATIVE)
INDICATORS_ABSOLUTE[18] = "Per capita GDP"
INDICATORS_ABSOLUTE[19] = "Per capita disposable income"

COST_INDICATORS = {
    "Per capita CO2 emissions",
    "CO2 emission intensity",
    "Energy consumption intensity",
    "Energy consumption structure",
    "Industrial smoke and dust emission intensity",
    "Industrial SO2 emission intensity",
    "Industrial wastewater emission intensity",
    "Fertilizer application intensity",
    "Per capita built-up area",
    "Unemployment rate",
}

SUBSYSTEMS_COMMON = {
    "CR": [
        "Per capita CO2 emissions",
        "CO2 emission intensity",
        "Energy consumption intensity",
        "Energy consumption structure",
    ],
    "PC": [
        "Industrial smoke and dust emission intensity",
        "Industrial SO2 emission intensity",
        "Industrial wastewater emission intensity",
        "Industrial solid waste treatment rate",
        "Fertilizer application intensity",
        "Environmental regulation intensity",
        "Environmental governance intensity",
    ],
    "GE": [
        "Green invention patent density",
        "Green utility model patent density",
        "Per capita afforestation area",
        "Forest coverage rate",
        "Green coverage rate of built-up areas",
        "Per capita park green space area",
        "Per capita built-up area",
    ],
}
SCORE_COLUMNS = {
    "CR": "Carbon reduction score",
    "PC": "Pollution control score",
    "GE": "Greening score",
    "GR": "Growth score",
}
SUBSYSTEM_LABELS = {
    "CR": "Carbon reduction",
    "PC": "Pollution control",
    "GE": "Greening",
    "GR": "Growth",
}
PARAMETERS = {
    "neutral_element": 1 / 26,
    "phi": 0.5,
    "model_weights": (1 / 3, 1 / 3, 1 / 3),
}


def paper_indicators(df: pd.DataFrame) -> list[str]:
    if "Per capita GDP growth rate" in df.columns:
        return list(INDICATORS_RELATIVE)
    return list(INDICATORS_ABSOLUTE)


def subsystems(df: pd.DataFrame) -> dict[str, list[str]]:
    out = {k: list(v) for k, v in SUBSYSTEMS_COMMON.items()}
    if "Per capita GDP growth rate" in df.columns:
        growth = ["Per capita GDP growth rate", "Per capita disposable income growth rate"]
    else:
        growth = ["Per capita GDP", "Per capita disposable income"]
    out["GR"] = growth + [
        "Financial development level",
        "Social consumption level",
        "External openness level",
        "Number of public buses per 10,000 people",
        "Urbanization level",
        "Unemployment rate",
    ]
    return out


def prepare_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year growth rates; 2011 is back-cast with the 2012 value."""
    out = df.sort_values(["Province", "Year"]).reset_index(drop=True).copy()
    for level, growth in [
        ("Per capita GDP", "Per capita GDP growth rate"),
        ("Per capita disposable income", "Per capita disposable income growth rate"),
    ]:
        g = out.groupby("Province")[level].pct_change() * 100
        g = g.groupby(out["Province"]).transform(lambda x: x.fillna(x.iloc[1]))
        out[growth] = g.values
    out = out.drop(columns=["Per capita GDP", "Per capita disposable income"])
    return out.sort_values(["Year", "Province"]).reset_index(drop=True)


def recover_raw(normalized: pd.DataFrame, minmax: pd.DataFrame) -> pd.DataFrame:
    """Recover raw growth-rate values from the normalized workbook."""
    df = normalized.copy()
    for _, row in minmax.iterrows():
        name = row["Indicator"]
        lo, hi = float(row["Min"]), float(row["Max"])
        if row["Direction"] == "Cost":
            df[name] = hi - df[name].astype(float) * (hi - lo)
        else:
            df[name] = lo + df[name].astype(float) * (hi - lo)
    return df


def read_panel(path: str) -> pd.DataFrame:
    """Read a raw panel or the normalized workbook (Normalized + MinMax sheets)."""
    sheets = pd.read_excel(path, sheet_name=None)
    if "Normalized" in sheets and "MinMax" in sheets:
        return recover_raw(sheets["Normalized"], sheets["MinMax"])
    df = next(iter(sheets.values()))
    return prepare_growth_rates(df)


def minmax_normalize(X: np.ndarray, indicators, cost) -> np.ndarray:
    cost = set(cost)
    Y = np.zeros_like(X, dtype=float)
    for j, name in enumerate(indicators):
        x = X[:, j]
        lo, hi = np.nanmin(x), np.nanmax(x)
        if hi == lo:
            Y[:, j] = 0.0
        elif name in cost:
            Y[:, j] = (hi - x) / (hi - lo)
        else:
            Y[:, j] = (x - lo) / (hi - lo)
    return Y


def entropy_weights(Y: np.ndarray) -> np.ndarray:
    m = Y.shape[0]
    P = Y / Y.sum(axis=0, keepdims=True)
    P = np.where(P <= 0, 1e-300, P)
    E = -1.0 / np.log(m) * (P * np.log(P)).sum(axis=0)
    return (1.0 - E) / (1.0 - E).sum()


def critic_weights(Y: np.ndarray) -> np.ndarray:
    S = Y.std(axis=0, ddof=1)
    R = np.corrcoef(Y, rowvar=False)
    A = (1.0 - R).sum(axis=0)
    C = S * A
    return C / C.sum()


def uninorm_combine(we: np.ndarray, wc: np.ndarray, g: float) -> np.ndarray:
    out = np.empty_like(we, dtype=float)
    for i, (e, c) in enumerate(zip(we, wc)):
        if e <= g and c <= g:
            out[i] = e * c / g
        elif e >= g and c >= g:
            out[i] = (e + c - e * c - g) / (1.0 - g)
        else:
            out[i] = (e + c) / 2.0
    return out / out.sum()


def compute_weights(df: pd.DataFrame) -> pd.DataFrame:
    inds = paper_indicators(df)
    X = df[inds].astype(float).to_numpy()
    Y = minmax_normalize(X, inds, COST_INDICATORS)
    we = entropy_weights(Y)
    wc = critic_weights(Y)
    w = uninorm_combine(we, wc, PARAMETERS["neutral_element"])
    return pd.DataFrame({
        "Indicator": inds,
        "Entropy weight": we,
        "CRITIC weight": wc,
        "Combined weight": w,
    })


def altrank(values: np.ndarray) -> np.ndarray:
    ordered = -np.sort(-values)
    rank = np.zeros(len(values), dtype=int)
    for i, v in enumerate(values):
        rank[i] = np.where(ordered == v)[0][0] + 1
    return rank


def dnma_components(data: pd.DataFrame, indicators, weights):
    X = data[list(indicators)].astype(float).to_numpy()
    m, n = X.shape
    x1 = np.zeros_like(X)
    x2 = np.zeros_like(X)
    for j, name in enumerate(indicators):
        x = X[:, j]
        target = x.min() if name in COST_INDICATORS else x.max()
        x1[:, j] = 1.0 - np.abs(x - target) / (max(x.max(), target) - min(x.min(), target))
        x2[:, j] = 1.0 - np.abs(x - target) / np.sqrt((x ** 2).sum() + target ** 2)
    u1 = (weights * (x1 / x1.max(axis=0))).sum(axis=1)
    u2 = (weights * (x1 / x1.max(axis=0))).min(axis=1)
    u3 = np.prod((x2 / x2.max(axis=0)) ** weights, axis=1)
    r1, r2, r3 = altrank(u1), altrank(u2), altrank(u3)
    phi = PARAMETERS["phi"]
    w1, w2, w3 = PARAMETERS["model_weights"]
    score = (
        w1 * np.sqrt(phi * (u1 / u1.max()) ** 2 + (1 - phi) * ((m - r1 + 1) / m) ** 2)
        + w2 * np.sqrt(phi * (u2 / u2.max()) ** 2 + (1 - phi) * ((m - r2 + 1) / m) ** 2)
        + w3 * np.sqrt(phi * (u3 / u3.max()) ** 2 + (1 - phi) * ((m - r3 + 1) / m) ** 2)
    )
    return score, u1, u2, u3


def compute_dnma(df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    subs = subsystems(df)
    wmap = dict(zip(weights["Indicator"], weights["Combined weight"]))
    out = df[["Province", "Year"]].copy()
    for key, inds in subs.items():
        w = np.array([wmap[i] for i in inds])
        scores = []
        for year in sorted(df["Year"].unique()):
            block = df[df["Year"] == year]
            scores.append(dnma_components(block, inds, w)[0])
        out[SCORE_COLUMNS[key]] = np.concatenate(scores)
    return out


def compute_balance(df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    subs = subsystems(df)
    wmap = dict(zip(weights["Indicator"], weights["Combined weight"]))
    rows = []
    for key, inds in subs.items():
        w = np.array([wmap[i] for i in inds])
        for year in sorted(df["Year"].unique()):
            block = df[df["Year"] == year].reset_index(drop=True)
            _, u1, u2, _ = dnma_components(block, inds, w)
            for province, cv, uv in zip(block["Province"], u1, u2):
                rows.append({
                    "Province": province,
                    "Year": year,
                    "Subsystem": SUBSYSTEM_LABELS[key],
                    "Complete compensatory score": cv,
                    "Un-compensatory score": uv,
                    "Difference": cv - uv,
                })
    return pd.DataFrame(rows)


def normalize_panel(df: pd.DataFrame) -> pd.DataFrame:
    inds = paper_indicators(df)
    out = df[["Province", "Year"] + inds].copy()
    for name in inds:
        x = df[name].astype(float)
        lo, hi = x.min(), x.max()
        if hi == lo:
            out[name] = 0.0
        elif name in COST_INDICATORS:
            out[name] = (hi - x) / (hi - lo)
        else:
            out[name] = (x - lo) / (hi - lo)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Subsystem calculation")
    parser.add_argument("--input", default=os.path.join(ROOT, "data", "panel_529.xlsx"))
    parser.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = parser.parse_args()

    df = read_panel(args.input)
    os.makedirs(args.outdir, exist_ok=True)

    normalize_panel(df).to_excel(os.path.join(args.outdir, "panel_529_normalized.xlsx"), index=False)
    weights = compute_weights(df)
    weights.to_excel(os.path.join(args.outdir, "weights.xlsx"), index=False)
    scores = compute_dnma(df, weights)
    scores.to_excel(os.path.join(args.outdir, "subsystem_scores.xlsx"), index=False)
    balance = compute_balance(df, weights)
    balance.to_excel(os.path.join(args.outdir, "subsystem_balance.xlsx"), index=False)

    print("Saved:", os.path.join(args.outdir, "weights.xlsx"))
    print("Saved:", os.path.join(args.outdir, "subsystem_scores.xlsx"))
    print("Saved:", os.path.join(args.outdir, "subsystem_balance.xlsx"))
    print("Saved:", os.path.join(args.outdir, "panel_529_normalized.xlsx"))


if __name__ == "__main__":
    main()
