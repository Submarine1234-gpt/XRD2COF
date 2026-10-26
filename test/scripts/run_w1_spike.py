"""scripts/run_w1_spike.py — W1 可行性尖刀 v2.1（全量自洽重发）

与之前版本的关键差别：
  1. 候选排序不再读 bb.get_X_points() —— pcb 建胞会就地变异 BuildingBlock，
     建胞后读到的是脏数据（曾出现 9 个 X 点）。改用 registry 的 CONNECTIVITY。
  2. 真值 c 用相对原型扰动 c_true = c_proto * U(0.94, 1.06)，
     不再假定层间距绝对值（避免把双层胞压成单层间距）。
  3. 误差指标用相对 c 误差(%)，跨 net / 堆积可比。
  4. 候选结构在所有 trial 之间只建一次。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from builder_bridge.pcb_adapter import (BBHandle, build_framework,
                                        framework_to_structure, import_pcb,
                                        load_building_block, quiet,
                                        set_c_parameter)
from config.topology_registry import CONNECTIVITY, resolve_nets, stacking_candidates
from pxrd_engine.forward_scorer import (Candidate, ScoreConfig,
                                        build_reflection_table,
                                        preprocess_experimental_pxrd,
                                        rank_candidates, diagnose, standard_grid,
                                        TWO_THETA_STEP, _two_theta_from_ginv,
                                        _structure_factor_sq, _lp_factor,
                                        _march_dollase, _po_cos_alpha)

CASES: Dict[str, Dict] = {
    # 3,2-c 蜂窝。net 唯一 -> 测 c 拟合 + 三种堆积判别。
    # _OH 变体被排除：0.0.9.4 把羟基 H 摆到邻分子 C-H 上（H...H = 1.055 A）。
    "tapb_pda": dict(sym1="T3", conn1="NH2", sym2="L2", conn2="CHO",
                     names1=("T3_BENZ_NH2_H", "T3_BENZ_NH2_OH"),
                     names2=("L2_BENZ_CHO_H",), blacklist=("_OH",)),
    # 4,2-c。PHPR 是实测唯一两个 net 都能在 AA 下建成的基元 -> 测 net 判别。
    # 318 原子，较慢，建议 --n-trials 3。
    "s4_l2": dict(sym1="S4", conn1="CHO", sym2="L2", conn2="NH2",
                  names1=("S4_PORP_CHO", "S4_PHPR_CHO", "S4_PTCA_CHO"),
                  names2=("L2_BENZ_NH2_H",), blacklist=("BTC",)),
}


def _filter(names, blacklist) -> List[str]:
    kept = [n for n in names if not any(b in n for b in (blacklist or ()))]
    return kept or list(names)


def verify_name(name: str, conn: str) -> bool:
    try:
        with quiet():
            bb = import_pcb().BuildingBlock(name=name, conector=conn)
        return int(getattr(bb, "n_atoms", 0) or 0) > 0
    except Exception:
        return False


def pick_bbs(spec: Dict):
    """选第一对真能建胞的基元；连接数顺序不合规时交换。"""
    sym1, sym2 = spec["sym1"], spec["sym2"]
    names1 = [n for n in _filter(spec["names1"], spec.get("blacklist"))
              if verify_name(n, spec["conn1"])]
    names2 = [n for n in _filter(spec["names2"], spec.get("blacklist"))
              if verify_name(n, spec["conn2"])]
    if not names1 or not names2:
        raise RuntimeError(f"基元名为空 after 校验: {names1} / {names2}")

    if CONNECTIVITY[sym1] < CONNECTIVITY[sym2]:
        print("[ORDER] BB1 连接数 < BB2，交换 A/B")
        sym1, sym2 = sym2, sym1
        names1, names2 = names2, names1
        spec = dict(spec, conn1=spec["conn2"], conn2=spec["conn1"])

    for n1 in names1:
        for n2 in names2:
            try:
                bb1 = load_building_block(BBHandle(n1, spec["conn1"], "builtin",
                                                   symmetry=sym1))
                bb2 = load_building_block(BBHandle(n2, spec["conn2"], "builtin",
                                                   symmetry=sym2))
                return bb1, bb2, n1, n2, sym1, sym2
            except Exception as exc:
                print(f"  skip {n1}+{n2}: {str(exc)[:90]}")
    raise RuntimeError("没有可用基元对")


def simulate(struct, c_true: float, po_r: float, disp: float,
             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """合成实验谱：与评分器不同构（高斯峰形 + 杂峰 + 噪声 + 非晶鼓包）。"""
    table = build_reflection_table(struct)
    lam = (1.0, 1.0, c_true / table.lattice0[2])
    d, tt = _two_theta_from_ginv(table.ginv, table.hkl, lam, table.wavelength)
    f2 = _structure_factor_sq(table.phase, table.atm_a, table.atm_b, table.atm_c, d)
    inten = f2 * table.mult * _lp_factor(tt) * \
        _march_dollase(_po_cos_alpha(table, lam), po_r)
    tt = tt - disp * np.cos(np.radians(tt / 2.0))

    grid = standard_grid()
    y = np.zeros_like(grid)
    fw_in = rng.uniform(0.30, 0.85)
    fw_st = rng.uniform(1.30, 3.20)
    for p, I, is_st in zip(tt, inten, table.is_stack):
        if not (0.5 < p < 36.0) or I <= 0:
            continue
        fw = fw_st if is_st else fw_in
        sig = fw / 2.3548
        lo = max(int((p - 4 * fw - grid[0]) / TWO_THETA_STEP), 0)
        hi = min(int((p + 4 * fw - grid[0]) / TWO_THETA_STEP) + 2, grid.size)
        if hi > lo:
            y[lo:hi] += I * np.exp(-0.5 * ((grid[lo:hi] - p) / sig) ** 2)

    for _ in range(2):                     # 未反应单体杂峰
        p0 = rng.uniform(5.0, 28.0)
        w = rng.uniform(0.35, 0.70)
        y += rng.uniform(0.05, 0.20) * np.exp(-0.5 * ((grid - p0) / (w / 2.3548)) ** 2)
    y += 0.06 * np.exp(-grid / 6.0)        # 低角非晶鼓包
    y += np.polyval([rng.uniform(-2e-4, 2e-4), 0.0, rng.uniform(0.01, 0.05)], grid)
    y = y / (float(np.max(y)) or 1.0)
    snr = rng.uniform(15.0, 50.0)
    return grid, np.clip(y + rng.normal(0.0, 1.0 / snr, y.size), 0.0, None)


def run_case(case_name: str, args) -> Dict:
    spec0 = CASES[case_name]
    rng = np.random.default_rng(args.seed)

    bb1, bb2, n1, n2, sym1, sym2 = pick_bbs(spec0)
    nets = resolve_nets(sym1, sym2)
    print(f"[BUILD] {n1} + {n2}  ({sym1}+{sym2})  nets={[n.net_id for n in nets]}")

    # ---- 候选结构只建一次 ----
    cands: List[Candidate] = []
    t_build = time.time()
    for net in nets:
        for st in stacking_candidates(net):
            try:
                s = framework_to_structure(
                    build_framework(bb1, bb2, net.net_token, st), check_geometry=False)
                cands.append(Candidate(net.net_id, net.net_token, st, s))
            except Exception as exc:
                print(f"  [skip] {net.net_id}/{st}: {str(exc)[:80]}")
    if not cands:
        raise RuntimeError("没有可建胞的候选")
    print(f"[CAND] {len(cands)} 个候选，建胞耗时 {time.time()-t_build:.1f}s")
    for c in cands:
        print(f"       {c.net_id}/{c.stacking:4s} a={c.structure.lattice.a:6.2f} "
              f"b={c.structure.lattice.b:6.2f} c={c.structure.lattice.c:5.2f} "
              f"atoms={len(c.structure)}")

    cfg = ScoreConfig()
    trials: List[Dict] = []
    for t in range(args.n_trials):
        cand = cands[int(rng.integers(len(cands)))]
        net = next(n for n in nets if n.net_id == cand.net_id)
        stacking = cand.stacking

        c_proto = cand.structure.lattice.c
        c_true = float(c_proto * rng.uniform(0.94, 1.06))
        po_r = float(rng.uniform(0.35, 1.0))
        disp = float(rng.uniform(-0.08, 0.08))

        grid, yy = simulate(cand.structure, c_true, po_r, disp, rng)
        exp = preprocess_experimental_pxrd(grid, yy)

        t0 = time.time()
        top, ok, bad = rank_candidates(cands, exp, cfg, top_k=3)
        dt = time.time() - t0
        if bad and not ok:
            for b in bad:
                print(f"      [ERR] {b['net_id']}/{b['stacking']}: {b.get('error')}")
            raise RuntimeError("全部候选打分失败，见上面 [ERR]（先跑 diagnose_score.py）")

        truth = f"{net.net_id}|{stacking}"
        hit = [f"{s['net_id']}|{s['stacking']}" for s in top if s.get("error") is None]
        trow = next((s for s in ok if f"{s['net_id']}|{s['stacking']}" == truth), None)
        rel_c = (abs(trow["c_fit"] - c_true) / c_true) if trow else float("nan")
        gap = ((top[0]["rwp"] - trow["rwp"])
               if (top and trow and top[0].get("error") is None) else float("nan"))

        trials.append(dict(truth=truth, top=hit, ok=ok,
                           r1=int(bool(hit) and hit[0] == truth),
                           r3=int(truth in hit),
                           rel_c=float(rel_c), gap=float(gap),
                           rwp_best=float(top[0]["rwp"]) if top else float("nan"),
                           dt=dt, n_cand=len(cands), c_true=c_true))
        print(f"  [{t+1}/{args.n_trials}] truth={truth:16s} "
              f"top1={hit[0] if hit else '--':16s} "
              f"Rwp={trials[-1]['rwp_best']*100:5.1f}% "
              f"relΔc={rel_c*100:5.2f}%  ({dt*1000:.0f}ms)")

    r1 = float(np.mean([x["r1"] for x in trials])) if trials else 0.0
    r3 = float(np.mean([x["r3"] for x in trials])) if trials else 0.0
    rc_vals = [x["rel_c"] for x in trials if np.isfinite(x["rel_c"])]
    gp_vals = [x["gap"] for x in trials if np.isfinite(x["gap"])]
    mrc = float(np.mean(rc_vals)) if rc_vals else float("nan")
    mgap = float(np.mean(gp_vals)) if gp_vals else float("nan")
    mt = float(np.mean([x["dt"] for x in trials])) if trials else float("nan")

    verdict = "GO" if (r3 >= 0.60 and np.isfinite(mrc) and mrc <= 0.02) \
        else "NO-GO / 需先修候选集或前向模型"
    summary = dict(case=case_name, n_trials=len(trials), n_candidates=len(cands),
                   bb=(n1, n2), nets=[n.net_id for n in nets],
                   recall_at_1=round(r1, 3), recall_at_3=round(r3, 3),
                   mean_rel_c_err=round(mrc, 4),
                   mean_rwp_gap=round(mgap, 4),
                   mean_seconds_per_trial=round(mt, 2),
                   verdict=verdict)
    print("\n" + "=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 70)
    if trials and trials[0]["ok"]:
        print(diagnose(trials[0]["ok"][0], exp))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="COF-PXRD2Struct W1 尖刀 v2.1")
    ap.add_argument("--case", default="tapb_pda", choices=list(CASES))
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", default="w1_spike_report.json")
    ap.add_argument("--list-topologies", action="store_true")
    args = ap.parse_args()

    if args.list_topologies:
        from config.topology_registry import export_table
        print(json.dumps(export_table(), ensure_ascii=False, indent=2))
        return 0

    try:
        summary = run_case(args.case, args)
    except Exception:
        traceback.print_exc()
        return 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {args.out}")
    return 0 if summary["verdict"].startswith("GO") else 2


if __name__ == "__main__":
    raise SystemExit(main())
