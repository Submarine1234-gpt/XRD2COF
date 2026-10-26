"""scripts/apply_patches.py — 幂等补丁器（只用函数边界切片，不用 str.replace）

前几轮静默失败的根因：str.replace 锚点差一个空格即无效，且无痕迹。
本脚本全部改成「按函数起始标记切片」，并在最后 import 自检真实行为。

用法：
    python scripts/apply_patches.py --check
    python scripts/apply_patches.py --apply
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FS = ROOT / "pxrd_engine" / "forward_scorer.py"
REG = ROOT / "config" / "topology_registry.py"


SCORECONFIG_NEW = '''class ScoreConfig:
    # c 的绝对界（Å）；None = 用相对原型的 c_rel_bounds。
    # 单层 AA 原型 c≈3.6，AAl/AAt 双层原型 c≈7.2 —— 绝对值会把后者压死。
    c_bounds: Optional[Tuple[float, float]] = None
    c_rel_bounds: Tuple[float, float] = (0.88, 1.14)
    lam_bounds: Tuple[float, float] = (0.95, 1.05)
    zero_bounds: Tuple[float, float] = (-0.15, 0.15)
    fwhm_inplane: Tuple[float, float] = (0.20, 1.20)
    fwhm_stack: Tuple[float, float] = (0.80, 4.00)
    disp_bounds: Tuple[float, float] = (-0.15, 0.15)
    po_r_bounds: Tuple[float, float] = (0.30, 1.00)
    eta: float = 0.50
    max_obs_peaks: int = 12
    match_tol_deg: float = 0.30
    rwp_weighting: str = "unit"


'''


FIT_NEW = '''def _c_bounds(cfg: "ScoreConfig", c0: float) -> Tuple[float, float]:
    """给定原型 c0，返回该候选的 c 拟合区间。"""
    if cfg.c_bounds is not None:
        return float(cfg.c_bounds[0]), float(cfg.c_bounds[1])
    lo = max(float(c0) * cfg.c_rel_bounds[0], 2.0)
    hi = min(float(c0) * cfg.c_rel_bounds[1], 60.0)
    if not (lo < c0 < hi):                       # 保证初值严格在界内
        lo, hi = min(lo, c0 * 0.9), max(hi, c0 * 1.1)
    return lo, hi


def fit_lattice_metric(table: ReflectionTable, obs_tt: np.ndarray,
                       symmetry: str, cfg: ScoreConfig, *,
                       obs_int: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """仅拟合峰位（Le Bail 思路）：[λa, λb, λc, zero]。坐标不动，无键长畸变。"""
    if obs_tt.size == 0:
        raise ValueError("实验谱未检出任何峰，无法拟合晶胞")
    c0 = table.lattice0[2]
    w = np.ones_like(obs_tt)
    if obs_int is not None and obs_int.size == obs_tt.size and np.max(obs_int) > 0:
        w = np.sqrt(np.clip(obs_int / np.max(obs_int), 0.05, 1.0))

    def residuals(p: np.ndarray) -> np.ndarray:
        la, lb, lc = _project_lam(p, symmetry)
        _, tt = _two_theta_from_ginv(table.ginv, table.hkl, (la, lb, lc),
                                     table.wavelength)
        tt = tt - p[3] * np.cos(np.radians(tt / 2.0))
        srt = np.sort(tt)
        idx = np.clip(np.searchsorted(srt, obs_tt), 1, srt.size - 1)
        left, right = srt[idx - 1], srt[idx]
        near = np.where(np.abs(obs_tt - left) < np.abs(obs_tt - right), left, right)
        return np.clip(obs_tt - near, -cfg.match_tol_deg, cfg.match_tol_deg) * w

    lo_c, hi_c = _c_bounds(cfg, c0)
    p0 = np.array([1.0, 1.0, float(np.clip(c0, lo_c, hi_c)) / c0, 0.0])
    lb_v = np.array([cfg.lam_bounds[0], cfg.lam_bounds[1],
                     lo_c / c0, hi_c / c0, cfg.zero_bounds[0]])
    ub_v = np.array([cfg.lam_bounds[1], cfg.lam_bounds[1],
                     hi_c / c0, hi_c / c0, cfg.zero_bounds[1]])
    p0 = np.clip(p0, lb_v + 1e-9, ub_v - 1e-9)

    res = least_squares(residuals, p0, bounds=(lb_v, ub_v),
                        loss="soft_l1", f_scale=0.05,
                        xtol=1e-8, ftol=1e-8, max_nfev=200)
    la, lb, lc = _project_lam(res.x, symmetry)
    return dict(lam_a=la, lam_b=lb, lam_c=lc, c=lc * c0,
                zero=float(res.x[3]), cost=float(res.cost),
                n_obs=int(obs_tt.size))


'''


SCORE_CAND_NEW = '''def score_candidate(cand: Candidate, exp: ExpPattern, cfg: ScoreConfig, *,
                    build_time: Optional[float] = None,
                    symmetry: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(net_id=cand.net_id, stacking=cand.stacking,
                               build_time=build_time, error=None)
    try:
        table = build_reflection_table(cand.structure)
        sym = symmetry or symmetry_of(cand.structure)
        mfit = fit_lattice_metric(table, exp.peaks_tt, sym, cfg,
                                  obs_int=exp.peaks_int)
        pfit = fit_profile_and_score(cand.structure, table, exp, mfit, cfg)
        out.update(metric=mfit,
                   profile={k: v for k, v in pfit.items() if k != "calc"},
                   rwp=pfit["rwp"], cos_sim=pfit["cos_sim"],
                   symmetry=sym, c_fit=mfit["c"])
        out["geo_compliance"] = float(max(abs(mfit["lam_a"] - 1.0),
                                          abs(mfit["lam_b"] - 1.0)))
        lo_c, hi_c = _c_bounds(cfg, cand.structure.lattice.c)
        out["c_at_bound"] = bool(mfit["c"] <= lo_c + 5e-3
                                 or mfit["c"] >= hi_c - 5e-3)
    except Exception as exc:
        out.update(error=f"{type(exc).__name__}: {exc}", rwp=float("inf"),
                   cos_sim=-1.0, geo_compliance=float("inf"), c_fit=float("nan"),
                   c_at_bound=False)
    return out


'''


def replace_block(src: str, start: str, end: str, new: str):
    i = src.find(start)
    if i < 0:
        return None, f"找不到起点 {start!r}"
    j = src.find(end, i + len(start))
    if j < 0:
        return None, f"找不到终点 {end!r}"
    return src[:i] + new + src[j:], "ok"


def check_state() -> int:
    src = FS.read_text(encoding="utf-8")
    ok = True
    checks = [
        ("c_bounds=None", "c_bounds: Optional[Tuple[float, float]] = None"),
        ("c_rel_bounds", "c_rel_bounds: Tuple[float, float] = (0.88, 1.14)"),
        ("_c_bounds()", "def _c_bounds("),
        ("c_at_bound 用 _c_bounds", "lo_c, hi_c = _c_bounds(cfg, cand.structure.lattice.c)"),
        ("旧 c_bounds 已消失", "c_bounds: Tuple[float, float] = (3.15, 4.25)"),
    ]
    for label, needle in checks:
        present = needle in src
        want = (label != "旧 c_bounds 已消失")
        good = present == want
        ok &= good
        print(f"  [{'OK ' if good else 'BAD'}] {label}")
    reg = REG.read_text(encoding="utf-8")
    print(f"  [{'OK ' if 'frozenset()' in reg else '-- '}] UPSTREAM_BLOCKED 已清空")
    return 0 if ok else 1


def apply_all() -> int:
    src = FS.read_text(encoding="utf-8")

    for tag, (start, end, new) in {
        "ScoreConfig": ("class ScoreConfig:", "def _poly_baseline(", SCORECONFIG_NEW),
        "fit_lattice_metric": ("def fit_lattice_metric(", "def _rwp(", FIT_NEW),
        "score_candidate": ("def score_candidate(", "def rank_candidates(", SCORE_CAND_NEW),
    }.items():
        out, msg = replace_block(src, start, end, new)
        if out is None:
            print(f"  !! {tag}: {msg}")
            return 2
        src = out
        print(f"  [APPLIED] {tag}")
    FS.write_text(src, encoding="utf-8")

    reg = REG.read_text(encoding="utf-8")
    old = 'UPSTREAM_BLOCKED: FrozenSet[str] = frozenset({"fxt_a"})'
    new = ('UPSTREAM_BLOCKED: FrozenSet[str] = frozenset()  '
           '# 0.0.9.4 复测：FXT_A 对 PORP/PTCA/TBPR/PHPR 可建胞')
    if old in reg:
        reg = reg.replace(old, new, 1)
        REG.write_text(reg, encoding="utf-8")
        print("  [APPLIED] UPSTREAM_BLOCKED -> frozenset()")
    elif "UPSTREAM_BLOCKED" not in reg:
        print("  [SKIP] registry 无 UPSTREAM_BLOCKED")
    else:
        print("  [SKIP] registry 已是空集")

    import ast
    ast.parse(FS.read_text(encoding="utf-8"))
    ast.parse(REG.read_text(encoding="utf-8"))
    print("  语法 OK")
    return 0


def verify() -> int:
    """import 后检查真实行为，而不只是文本。"""
    sys.path.insert(0, str(ROOT))
    from pxrd_engine.forward_scorer import ScoreConfig, _c_bounds
    from config.topology_registry import resolve_nets
    cfg = ScoreConfig()
    print("  c_bounds     =", cfg.c_bounds)
    print("  c_rel_bounds =", cfg.c_rel_bounds)
    print("  _c_bounds(3.6) =", tuple(round(x, 3) for x in _c_bounds(cfg, 3.6)))
    print("  _c_bounds(7.2) =", tuple(round(x, 3) for x in _c_bounds(cfg, 7.2)))
    print("  S4+L2 nets     =", [n.net_id for n in resolve_nets("S4", "L2")])
    ok = (cfg.c_bounds is None
          and abs(_c_bounds(cfg, 3.6)[0] - 3.168) < 1e-6
          and abs(_c_bounds(cfg, 7.2)[1] - 8.208) < 1e-6)
    print("  ==>", "PASS" if ok else "FAIL")
    return 0 if ok else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.check:
        print("== 当前状态 ==")
        return check_state()
    if args.apply:
        print("== 应用补丁 ==")
        rc = apply_all()
        if rc:
            return rc
        print("\n== 行为自检 ==")
        return verify()
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
