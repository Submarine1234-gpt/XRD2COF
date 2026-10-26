"""scripts/diagnose_score.py — 把 score_candidate 吞掉的异常捞出来。"""
from __future__ import annotations
import os, sys, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from builder_bridge.pcb_adapter import (BBHandle, build_framework,
                                        framework_to_structure, load_building_block)
from config.topology_registry import resolve_nets
from pxrd_engine.forward_scorer import (
    ScoreConfig, build_reflection_table, fit_lattice_metric,
    fit_profile_and_score, preprocess_experimental_pxrd, symmetry_of,
    score_candidate, Candidate)
from scripts.run_w1_spike import simulate

STACK = sys.argv[1] if len(sys.argv) > 1 else "AA"

bb1 = load_building_block(BBHandle("T3_BENZ_NH2_H", "NH2", symmetry="T3"))
bb2 = load_building_block(BBHandle("L2_BENZ_CHO_H", "CHO", symmetry="L2"))
struct = framework_to_structure(build_framework(bb1, bb2, "HCB_A", STACK),
                               check_geometry=False)
print(f"struct: a={struct.lattice.a:.2f} c={struct.lattice.c:.2f} atoms={len(struct)}")

rng = np.random.default_rng(0)
c_true = struct.lattice.c * 1.03
grid, yy = simulate(struct, c_true, po_r=0.7, disp=0.03, rng=rng)
exp = preprocess_experimental_pxrd(grid, yy)
print(f"exp: n_peaks={exp.peaks_tt.size} peaks={np.round(exp.peaks_tt[:8],2)}")
print(f"     valid={int(exp.valid.sum())}/{exp.valid.size}  ymax={yy.max():.3f}")

print("\n--- 1. build_reflection_table ---")
try:
    table = build_reflection_table(struct)
    print(f"OK n_ref={table.hkl.shape[0]} 2θ={table.tt0.min():.2f}-{table.tt0.max():.2f}")
except Exception:
    traceback.print_exc(); raise SystemExit(1)

sym = symmetry_of(struct)
print(f"\n--- 2. fit_lattice_metric (sym={sym}, c0={struct.lattice.c:.2f}) ---")
try:
    mfit = fit_lattice_metric(table, exp.peaks_tt, sym, ScoreConfig(),
                              obs_int=exp.peaks_int)
    print("OK", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mfit.items()})
except Exception:
    traceback.print_exc(); raise SystemExit(2)

print("\n--- 3. fit_profile_and_score ---")
try:
    pfit = fit_profile_and_score(struct, table, exp, mfit, ScoreConfig())
    print("OK", {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in pfit.items() if k != "calc"})
except Exception:
    traceback.print_exc(); raise SystemExit(3)

print("\n--- 4. score_candidate（看被吞掉的 error 字段）---")
cand = Candidate("hcb_a", "HCB_A", STACK, struct)
out = score_candidate(cand, exp, ScoreConfig())
print("error =", out.get("error"))
if out.get("error") is None:
    m, pr = out["metric"], out["profile"]
    print(f"  Rwp={out['rwp']*100:.2f}%  cos={out['cos_sim']:.4f}")
    print(f"  c={m['c']:.3f}  λa={m['lam_a']:.4f}  λb={m['lam_b']:.4f}  "
          f"zero={m['zero']:+.3f}  n_eval={m['n_eval']}")
    print(f"  fwhm_in={pr['fwhm_inplane']:.3f}  fwhm_st={pr['fwhm_stack']:.3f}  "
          f"disp={pr['disp']:+.3f}  PO={pr['po_r']:.3f}")
    print(f"  c_at_bound={out['c_at_bound']}  geo={out['geo_compliance']*100:.2f}%")
