"""scripts/apply_patches2.py — 修 fit_lattice_metric 的 4/5 维 bug + 峰值提取

幂等：全部按函数边界切片重写，不用 str.replace 锚点。
"""
from __future__ import annotations
import ast, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FS = ROOT / "pxrd_engine" / "forward_scorer.py"

FIT_NEW = '''def fit_lattice_metric(table: ReflectionTable, obs_tt: np.ndarray,
                       symmetry: str, cfg: ScoreConfig, *,
                       obs_int: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """仅拟合峰位（Le Bail 思路）：4 个自由参数 [λa, λb, λc, zero]。

    注意：参数是 4 维，lb_v / ub_v 也必须是 4 维（曾经误写成 5 维导致
    np.clip 广播失败，异常被 score_candidate 吞掉，表现为 Rwp=inf）。
    """
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
    lc0 = float(np.clip(c0, lo_c * 1.001, hi_c * 0.999)) / c0
    p0 = np.array([1.0, 1.0, 1.0, 0.0])
    lb_v = np.array([cfg.lam_bounds[0], cfg.lam_bounds[0],
                     lo_c / c0, cfg.zero_bounds[0]])
    ub_v = np.array([cfg.lam_bounds[1], cfg.lam_bounds[1],
                     hi_c / c0, cfg.zero_bounds[1]])
    p0[2] = lc0
    p0 = np.clip(p0, lb_v + 1e-9, ub_v - 1e-9)

    res = least_squares(residuals, p0, bounds=(lb_v, ub_v),
                        loss="soft_l1", f_scale=0.05,
                        xtol=1e-8, ftol=1e-8, max_nfev=400)
    la, lb, lc = _project_lam(res.x, symmetry)
    return dict(lam_a=la, lam_b=lb, lam_c=lc, c=lc * c0,
                zero=float(res.x[3]), cost=float(res.cost),
                n_obs=int(obs_tt.size), n_eval=int(res.nfev))


'''

PREP_NEW = '''def preprocess_experimental_pxrd(two_theta: Sequence[float],
                                 intensity: Sequence[float], *,
                                 low_angle_cut: float = LOW_ANGLE_CUT,
                                 baseline_order: int = 6,
                                 n_peaks: int = 12,
                                 peak_tt_max: float = 32.0,
                                 peak_rel_height: float = 0.06) -> ExpPattern:
    """重采样 -> 遮蔽低角 -> 扣基线 -> 归一化 -> 寻峰 -> d 特征。

    寻峰改进（W1 实测驱动）：
      * peak_rel_height：峰值须 > 6% 最高峰，滤掉基线残余噪声；
      * peak_tt_max：>32° 的峰信息量低、噪声高，不参与晶胞拟合；
      * 输出按 2θ 升序，便于 d_features 与人工核对。
    """
    tt = np.asarray(two_theta, dtype=float).ravel()
    ii = np.asarray(intensity, dtype=float).ravel()
    if tt.size != ii.size or tt.size < 10:
        raise ValueError("PXRD 数据长度不一致或点数过少")
    order = np.argsort(tt)
    tt, ii = tt[order], ii[order]
    uniq, inv = np.unique(np.round(tt, 6), return_inverse=True)
    if uniq.size != tt.size:
        sums = np.bincount(inv, weights=ii)
        cnts = np.bincount(inv)
        tt, ii = uniq, sums / np.maximum(cnts, 1)

    grid = standard_grid()
    y = np.interp(grid, tt, ii, left=ii[0], right=ii[-1])
    y = y - _poly_baseline(grid, y, order=baseline_order)
    y = np.clip(y, 0.0, None)
    span = float(np.max(y)) or 1.0
    y = y / span
    ymax = float(np.max(y))

    valid = grid >= low_angle_cut
    pk, props = find_peaks(
        y,
        prominence=max(1e-3, 0.4 * peak_rel_height * ymax),
        height=peak_rel_height * ymax,
        distance=max(1, int(0.12 / TWO_THETA_STEP)),
    )
    if pk.size:
        pk = pk[grid[pk] <= peak_tt_max]
    if pk.size:
        order_pk = np.argsort(props["prominences"][:pk.size] if False else
                              y[pk])[::-1][:n_peaks]
        pk = np.sort(pk[order_pk])
        fwhm = np.array([_estimate_fwhm(grid, y, p) for p in pk])
        peaks_tt, peaks_int = grid[pk], y[pk]
    else:
        fwhm = np.array([])
        peaks_tt, peaks_int = np.array([]), np.array([])

    return ExpPattern(
        two_theta=grid, intensity=y, valid=valid,
        peaks_tt=peaks_tt, peaks_int=peaks_int, peaks_fwhm=fwhm,
        d_features=_d_from_two_theta(peaks_tt[:4]),
        meta=dict(n_peaks=int(pk.size), low_angle_cut=low_angle_cut,
                  peak_tt_max=peak_tt_max, peak_rel_height=peak_rel_height,
                  raw_points=int(tt.size), span=span),
    )


'''

CSENS = '''def c_resolvability(table: ReflectionTable, rel: float = 0.01) -> Dict[str, float]:
    """1% 的 c 变化能在最强 00l 峰上产生多大 2θ 位移（度）。

    < 0.05° 意味着 c 与 2θ 步长（0.02°）不可分辨，该候选的 c 不可信。
    """
    if not np.any(table.is_stack):
        return dict(delta_tt=0.0, best_tt=float("nan"),
                    note="2θ 范围内无 00l 反射 -> c 完全不可辨识")
    d0, tt0 = _two_theta_from_ginv(table.ginv, table.hkl, (1.0, 1.0, 1.0),
                                   table.wavelength)
    d1, tt1 = _two_theta_from_ginv(table.ginv, table.hkl, (1.0, 1.0, 1.0 + rel),
                                   table.wavelength)
    m = table.is_stack
    delta = np.abs(tt1[m] - tt0[m])
    k = int(np.argmax(delta))
    return dict(delta_tt=float(delta[k]),
                best_tt=float(tt0[m][k]),
                note="OK" if float(delta[k]) > 0.05 else "c 几乎不可辨识")


'''


def slice_rewrite(src: str, start: str, end_marker: str, new: str, tag: str) -> str:
    i = src.find(start)
    if i < 0:
        raise SystemExit(f"[{tag}] 找不到起点 {start!r}")
    j = src.find(end_marker, i + len(start))
    if j < 0:
        raise SystemExit(f"[{tag}] 找不到终点 {end_marker!r}")
    print(f"  [APPLIED] {tag}")
    return src[:i] + new + src[j:]


def main() -> int:
    src = FS.read_text(encoding="utf-8")

    if "lb_v = np.array([cfg.lam_bounds[0], cfg.lam_bounds[1]," in src:
        src = slice_rewrite(src, "def fit_lattice_metric(", "def _rwp(", FIT_NEW,
                            "fit_lattice_metric (4维边界)")
    else:
        print("  [SKIP] fit_lattice_metric 已是 4 维")

    if "peak_rel_height: float = 0.06" not in src:
        src = slice_rewrite(src, "def preprocess_experimental_pxrd(",
                            "def _cm_arrays(", PREP_NEW, "preprocess (峰值过滤)")
    else:
        print("  [SKIP] preprocess 已含峰值过滤")

    if "def c_resolvability(" not in src:
        anchor = "def build_reflection_table("
        k = src.find(anchor)
        src = src[:k] + CSENS + src[k:]
        print("  [APPLIED] c_resolvability")
    else:
        print("  [SKIP] c_resolvability 已存在")

    ast.parse(src)
    FS.write_text(src, encoding="utf-8")
    print("  语法 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
