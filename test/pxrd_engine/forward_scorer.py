"""pxrd_engine/forward_scorer.py

前向 PXRD 物理模型 + 峰位度量拟合 + Rwp 打分排序。

性能要点：结构因子只依赖分数坐标，与晶格度量无关。因此每个候选结构
只需预计算一次「相位矩阵 phase(MxN)」与「逆度量张量 6 分量」，之后
改变 [λa, λb, c] 只需几行向量运算即可得到全部 2θ 与 d：
    1/d² = h²G00/λa² + k²G11/λb² + l²G22/λc²
           + 2hk G01/(λaλb) + 2hl G02/(λaλc) + 2kl G12/(λbλc)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy.optimize import least_squares
from scipy.signal import find_peaks

WAVELENGTH_CU_KA = 1.5406
TWO_THETA_MIN = 1.5
TWO_THETA_MAX = 35.0
TWO_THETA_STEP = 0.02
LOW_ANGLE_CUT = 2.0

_CM_COEFFS = {
    "H":  ((0.493002, 0.322912, 0.140191, 0.040810),
           (10.5109, 26.1257, 3.14236, 57.7997), 0.003038),
    "B":  ((2.054500, 1.332600, 1.097900, 0.706800),
           (23.2185, 1.02100, 60.3498, 0.140300), -0.193200),
    "C":  ((2.310000, 1.020000, 1.588600, 0.865000),
           (20.8439, 10.2075, 0.568700, 51.6512), 0.215600),
    "N":  ((12.21260, 3.132200, 2.012500, 1.166300),
           (0.005700, 9.89330, 28.9975, 0.582600), -11.5290),
    "O":  ((3.048500, 2.286800, 1.546300, 0.867000),
           (13.2771, 5.70110, 0.323900, 32.9089), 0.250800),
    "F":  ((3.539200, 2.641200, 1.517000, 1.024300),
           (10.2825, 4.29440, 0.261500, 26.1476), 0.277600),
    "Si": ((6.291500, 3.035300, 1.989100, 1.541000),
           (2.43860, 32.3337, 0.678500, 81.6937), 1.140700),
    "P":  ((6.434500, 4.179100, 1.780000, 1.490800),
           (1.90690, 27.1570, 0.526000, 68.1645), 1.114900),
    "S":  ((6.905300, 5.203400, 1.437900, 1.586300),
           (1.46790, 22.2151, 0.253600, 56.1720), 0.866900),
    "Cl": ((11.46040, 7.196400, 6.255600, 1.645500),
           (0.010400, 1.16620, 18.5194, 47.7784), -9.557400),
}
_CM_FALLBACK = (2.0, 1.5, 1.0, 0.6), (20.0, 8.0, 2.0, 0.5), 0.1


def standard_grid() -> np.ndarray:
    n = int(round((TWO_THETA_MAX - TWO_THETA_MIN) / TWO_THETA_STEP)) + 1
    return np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, n)


@dataclass
class ExpPattern:
    two_theta: np.ndarray
    intensity: np.ndarray
    valid: np.ndarray
    peaks_tt: np.ndarray
    peaks_int: np.ndarray
    peaks_fwhm: np.ndarray
    d_features: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionTable:
    hkl: np.ndarray
    d0: np.ndarray
    tt0: np.ndarray
    mult: np.ndarray
    ginv: np.ndarray
    f2_0: np.ndarray
    phase: np.ndarray
    atm_a: np.ndarray
    atm_b: np.ndarray
    atm_c: np.ndarray
    is_stack: np.ndarray
    wavelength: float
    lattice0: Tuple[float, float, float, float, float, float]


@dataclass
class ScoreConfig:
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


def _poly_baseline(x: np.ndarray, y: np.ndarray, order: int = 6,
                   n_iter: int = 8) -> np.ndarray:
    """迭代多项式基线：每轮只用低包络点重拟合（抗强峰拉抬）。"""
    m = np.ones_like(y, dtype=bool)
    base = np.zeros_like(y)
    for _ in range(n_iter):
        if m.sum() < order + 2:
            break
        coef = np.polyfit(x[m], y[m], order)
        base = np.polyval(coef, x)
        resid = y - base
        thr = np.percentile(resid[m], 55) + 0.5 * (np.std(resid[m]) or 1.0)
        new_m = resid <= thr
        if new_m.sum() < order + 2:
            break
        m = new_m
    return base


def _estimate_fwhm(x: np.ndarray, y: np.ndarray, idx: int) -> float:
    h = y[idx] / 2.0
    left = idx
    while left > 0 and y[left] > h:
        left -= 1
    right = idx
    while right < len(y) - 1 and y[right] > h:
        right += 1
    return float(max(x[right] - x[left], TWO_THETA_STEP))


def _d_from_two_theta(tt_deg, wavelength: float = WAVELENGTH_CU_KA) -> np.ndarray:
    th = np.radians(np.asarray(tt_deg, dtype=float) / 2.0)
    s = np.sin(th)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 1e-6, wavelength / (2.0 * s), np.nan)


def preprocess_experimental_pxrd(two_theta: Sequence[float],
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


def _cm_arrays(symbols: Sequence[str]):
    a = np.zeros((len(symbols), 4))
    b = np.zeros((len(symbols), 4))
    c = np.zeros(len(symbols))
    for i, s in enumerate(symbols):
        aa, bb, cc = _CM_COEFFS.get(s, _CM_FALLBACK)
        a[i], b[i], c[i] = aa, bb, cc
    return a, b, c


def _ginv_components(lattice: Lattice) -> np.ndarray:
    """逆度量张量的 6 个独立分量 (G00,G11,G22,G01,G02,G12)。"""
    Gi = np.linalg.inv(np.asarray(lattice.metric_tensor, dtype=float))
    return np.array([Gi[0, 0], Gi[1, 1], Gi[2, 2],
                     Gi[0, 1], Gi[0, 2], Gi[1, 2]], dtype=float)


def _q_from_ginv(ginv: np.ndarray, hkl: np.ndarray, lam: Sequence[float]) -> np.ndarray:
    la, lb, lc = lam
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    q = (h * h * ginv[0] / (la * la)
         + k * k * ginv[1] / (lb * lb)
         + l * l * ginv[2] / (lc * lc)
         + 2.0 * h * k * ginv[3] / (la * lb)
         + 2.0 * h * l * ginv[4] / (la * lc)
         + 2.0 * k * l * ginv[5] / (lb * lc))
    return np.clip(q, 1e-12, None)


def _two_theta_from_ginv(ginv, hkl, lam,
                         wavelength: float = WAVELENGTH_CU_KA):
    q = _q_from_ginv(ginv, hkl, lam)
    d = 1.0 / np.sqrt(q)
    s = wavelength / (2.0 * d)
    tt = 2.0 * np.degrees(np.arcsin(np.clip(s, 0.0, 1.0)))
    return d, tt


def _structure_factor_sq(phase: np.ndarray, atm_a, atm_b, atm_c,
                         d: np.ndarray) -> np.ndarray:
    """|F|² = |Σ_j f_j(s) · exp(2πi h·x_j)|²，向量化。"""
    s2 = 0.25 / np.square(np.clip(d, 1e-3, None))
    f_atoms = np.zeros(phase.shape, dtype=float)
    for k in range(4):
        f_atoms += atm_a[None, :, k] * np.exp(-atm_b[None, :, k] * s2[:, None])
    f_atoms += atm_c[None, :]
    F = np.einsum("mn,mn->m", phase, f_atoms)
    return np.abs(F) ** 2


def c_resolvability(table: ReflectionTable, rel: float = 0.01) -> Dict[str, float]:
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


def build_reflection_table(struct: Structure, *,
                           wavelength: float = WAVELENGTH_CU_KA,
                           tt_range: Tuple[float, float] = (0.8, 36.0),
                           merge_tol_deg: float = 0.02) -> ReflectionTable:
    """枚举全反射、按 2θ 合并近重合峰、预计算相位矩阵与度量分量。

    枚举上界由 hkl^T G* hkl <= 1/d_min^2 的对角项给出：
        |h| <= 1 / (d_min * sqrt(G*_00))
    这是严格上界，不会漏反射。
    """
    lat = struct.lattice
    a, b, c, al, be, ga = lat.parameters
    ginv = _ginv_components(lat)

    d_min = wavelength / (2.0 * np.sin(np.radians(tt_range[1] / 2.0))) * 0.98
    hmax = int(np.ceil(1.0 / (d_min * np.sqrt(ginv[0])))) + 1
    kmax = int(np.ceil(1.0 / (d_min * np.sqrt(ginv[1])))) + 1
    lmax = int(np.ceil(1.0 / (d_min * np.sqrt(ginv[2])))) + 1
    if (2 * hmax + 1) * (2 * kmax + 1) * (2 * lmax + 1) > 4_000_000:
        raise ValueError(f"反射枚举空间过大 (h<={hmax},k<={kmax},l<={lmax})")

    hh, kk, ll = np.meshgrid(np.arange(-hmax, hmax + 1),
                             np.arange(-kmax, kmax + 1),
                             np.arange(-lmax, lmax + 1), indexing="ij")
    hkl_all = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1).astype(np.int32)
    hkl_all = hkl_all[~np.all(hkl_all == 0, axis=1)]

    d_all, tt_all = _two_theta_from_ginv(ginv, hkl_all, (1.0, 1.0, 1.0), wavelength)
    keep = (tt_all >= tt_range[0]) & (tt_all <= tt_range[1])
    hkl_all, d_all, tt_all = hkl_all[keep], d_all[keep], tt_all[keep]
    if hkl_all.shape[0] == 0:
        raise ValueError("该晶胞在给定 2θ 范围内无反射")
    if hkl_all.shape[0] < 15:
        warnings.warn(
            f"反射数仅 {hkl_all.shape[0]} 个，晶胞可能过大或枚举范围过窄",
            RuntimeWarning, stacklevel=2)

    frac = np.asarray(struct.frac_coords, dtype=float)
    species = [str(s.symbol) for s in struct.species]
    atm_a, atm_b, atm_c = _cm_arrays(species)
    phase_all = np.exp(2j * np.pi * (hkl_all @ frac.T))
    f2_all = _structure_factor_sq(phase_all, atm_a, atm_b, atm_c, d_all)

    order = np.argsort(tt_all)
    tt_s = tt_all[order]
    breaks = np.where(np.diff(tt_s) > merge_tol_deg)[0] + 1
    groups = np.split(order, breaks)

    rep_idx = np.array([g[np.argmax(f2_all[g])] for g in groups], dtype=int)
    mult = np.array([float(len(g)) for g in groups])
    hkl = hkl_all[rep_idx]

    return ReflectionTable(
        hkl=hkl, d0=d_all[rep_idx], tt0=tt_all[rep_idx], mult=mult, ginv=ginv,
        f2_0=f2_all[rep_idx] * mult, phase=phase_all[rep_idx],
        atm_a=atm_a, atm_b=atm_b, atm_c=atm_c,
        is_stack=(hkl[:, 0] == 0) & (hkl[:, 1] == 0) & (hkl[:, 2] != 0),
        wavelength=wavelength, lattice0=(a, b, c, al, be, ga),
    )


def _lp_factor(tt_deg: np.ndarray) -> np.ndarray:
    th = np.radians(np.clip(tt_deg, 1e-3, 179.0) / 2.0)
    c2 = np.cos(2 * th)
    out = (1.0 + c2 * c2) / (np.sin(th) ** 2 * np.cos(th))
    return np.where(tt_deg > 0.5, out, 0.0)


def _march_dollase(cos_alpha: np.ndarray, r: float) -> np.ndarray:
    r = float(np.clip(r, 1e-3, None))
    ca2 = np.clip(cos_alpha, 0.0, 1.0) ** 2
    return (r * r * ca2 + (1.0 / r) * (1.0 - ca2)) ** (-1.5)


def _po_cos_alpha(table: ReflectionTable, lam: Sequence[float]) -> np.ndarray:
    """择优取向轴：c*（层法向）。由晶胞参数重建矩阵，与结构朝向无关。"""
    a0, b0, c0, al, be, ga = table.lattice0
    la, lb, lc = lam
    lat = Lattice.from_parameters(a0 * la, b0 * lb, c0 * lc, al, be, ga)
    recip = np.asarray(lat.reciprocal_lattice.matrix, dtype=float)
    g = table.hkl.astype(float) @ recip
    gn = np.linalg.norm(g, axis=1)
    u = recip[2] / (np.linalg.norm(recip[2]) or 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_a = np.abs(g @ u) / np.clip(gn, 1e-12, None)
    return np.clip(np.nan_to_num(cos_a), 0.0, 1.0)


def _pv(x: np.ndarray, x0: float, fwhm: float, eta: float = 0.5) -> np.ndarray:
    fwhm = max(float(fwhm), 0.02)
    sigma = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - x0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    lor = (fwhm / (2 * np.pi)) / ((x - x0) ** 2 + (fwhm / 2) ** 2)
    return eta * lor + (1.0 - eta) * g


def _accumulate_profile(grid: np.ndarray, positions: np.ndarray,
                        intensities: np.ndarray, fwhm: np.ndarray,
                        eta: float = 0.5) -> np.ndarray:
    """局部窗口累加伪 Voigt（比全广播快 5-10 倍）。"""
    y = np.zeros_like(grid)
    step = grid[1] - grid[0]
    imax = float(np.max(intensities)) if intensities.size else 0.0
    sel = ((positions > grid[0] - 3.0) & (positions < grid[-1] + 3.0)
           & (intensities > 1e-8 * imax))
    for p, I, w in zip(positions[sel], intensities[sel], fwhm[sel]):
        half = max(4.0 * max(w, step), 8 * step)
        i0 = int((p - half - grid[0]) / step)
        i1 = int((p + half - grid[0]) / step) + 2
        i0, i1 = max(i0, 0), min(i1, len(grid))
        if i1 > i0:
            y[i0:i1] += I * _pv(grid[i0:i1], p, w, eta)
    return y


def compute_pattern(struct: Structure, table: ReflectionTable, *,
                    c: Optional[float] = None,
                    lam_a: float = 1.0, lam_b: float = 1.0,
                    po_r: float = 1.0, disp: float = 0.0,
                    fwhm_inplane: float = 0.45, fwhm_stack: float = 1.80,
                    grid: Optional[np.ndarray] = None, eta: float = 0.5
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (grid, calc_intensity)。诊断/画图用。"""
    grid = standard_grid() if grid is None else grid
    lc = (c / table.lattice0[2]) if c is not None else 1.0
    lam = (lam_a, lam_b, lc)
    d, tt = _two_theta_from_ginv(table.ginv, table.hkl, lam, table.wavelength)
    f2 = _structure_factor_sq(table.phase, table.atm_a, table.atm_b, table.atm_c, d)
    inten = f2 * table.mult * _lp_factor(tt)
    if po_r != 1.0:
        inten = inten * _march_dollase(_po_cos_alpha(table, lam), po_r)
    tt = tt - disp * np.cos(np.radians(tt / 2.0))
    fw = np.where(table.is_stack, fwhm_stack, fwhm_inplane)
    return grid, _accumulate_profile(grid, tt, inten, fw, eta)


def _project_lam(p: np.ndarray, symmetry: str) -> Tuple[float, float, float]:
    la, lb, lc = float(p[0]), float(p[1]), float(p[2])
    if symmetry in ("hexagonal", "trigonal", "tetragonal", "cubic"):
        lb = la
    return la, lb, lc


def _c_bounds(cfg: "ScoreConfig", c0: float) -> Tuple[float, float]:
    """给定原型 c0，返回该候选的 c 拟合区间。"""
    if cfg.c_bounds is not None:
        return float(cfg.c_bounds[0]), float(cfg.c_bounds[1])
    lo = max(float(c0) * cfg.c_rel_bounds[0], 2.0)
    hi = min(float(c0) * cfg.c_rel_bounds[1], 60.0)
    if not (lo < c0 < hi):                       # 保证初值严格在界内
        lo, hi = min(lo, c0 * 0.9), max(hi, c0 * 1.1)
    return lo, hi


def _c_bounds(cfg: "ScoreConfig", c0: float) -> Tuple[float, float]:
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


def _rwp(y_obs: np.ndarray, y_calc: np.ndarray, w: np.ndarray) -> float:
    den = float(np.sum(w * y_obs * y_obs))
    if den <= 0:
        return float("inf")
    return float(np.sqrt(np.sum(w * (y_obs - y_calc) ** 2) / den))


def fit_profile_and_score(struct: Structure, table: ReflectionTable,
                          exp: ExpPattern, metric_fit: Dict[str, Any],
                          cfg: ScoreConfig) -> Dict[str, Any]:
    """固定度量后拟合 [fwhm_in, fwhm_stk, disp, po_r]，scale 取解析解。"""
    grid = exp.two_theta
    y_obs = exp.intensity
    w = (np.ones_like(y_obs) if cfg.rwp_weighting == "unit"
         else 1.0 / np.clip(y_obs, 1e-3, None))
    m = exp.valid
    y_o, w_m = y_obs[m], w[m]
    if y_o.size < 10:
        raise ValueError("有效数据点过少")

    lam = (metric_fit["lam_a"], metric_fit["lam_b"],
           metric_fit["c"] / table.lattice0[2])
    d, tt = _two_theta_from_ginv(table.ginv, table.hkl, lam, table.wavelength)
    f2 = _structure_factor_sq(table.phase, table.atm_a, table.atm_b, table.atm_c, d)
    base_static = f2 * table.mult * _lp_factor(tt)
    cos_alpha = _po_cos_alpha(table, lam)

    def model(x: np.ndarray) -> np.ndarray:
        f_in, f_st, disp, po_r = x
        inten = base_static * _march_dollase(cos_alpha, po_r)
        tt_s = tt - disp * np.cos(np.radians(tt / 2.0))
        fw = np.where(table.is_stack, f_st, f_in)
        return _accumulate_profile(grid, tt_s, inten, fw, cfg.eta)

    def residuals(x: np.ndarray) -> np.ndarray:
        y_c = model(x)[m]
        denom = float(np.sum(w_m * y_c * y_c))
        s = float(np.sum(w_m * y_o * y_c) / denom) if denom > 1e-12 else 0.0
        return np.sqrt(w_m) * (y_o - s * y_c)

    x0 = np.array([0.45, 1.80, 0.0, 0.80])
    lb_v = np.array([cfg.fwhm_inplane[0], cfg.fwhm_stack[0],
                     cfg.disp_bounds[0], cfg.po_r_bounds[0]])
    ub_v = np.array([cfg.fwhm_inplane[1], cfg.fwhm_stack[1],
                     cfg.disp_bounds[1], cfg.po_r_bounds[1]])
    x0 = np.clip(x0, lb_v + 1e-6, ub_v - 1e-6)
    res = least_squares(residuals, x0, bounds=(lb_v, ub_v),
                        xtol=1e-4, ftol=1e-4, max_nfev=80)

    y_c = model(res.x)
    den = float(np.sum(w_m * y_c[m] ** 2))
    scale = float(np.sum(w_m * y_o * y_c[m]) / den) if den > 1e-12 else 0.0
    y_c = y_c * scale
    rwp = _rwp(y_o, y_c[m], w_m)
    aa = y_c[m] - y_c[m].mean()
    bb = y_o - y_o.mean()
    cos_sim = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))

    return dict(rwp=rwp, cos_sim=cos_sim, scale=scale,
                fwhm_inplane=float(res.x[0]), fwhm_stack=float(res.x[1]),
                disp=float(res.x[2]), po_r=float(res.x[3]),
                calc=y_c, n_eval=int(res.nfev))


@dataclass
class Candidate:
    net_id: str
    net_token: str
    stacking: str
    structure: Structure
    meta: Dict[str, Any] = field(default_factory=dict)


def symmetry_of(struct: Structure) -> str:
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return SpacegroupAnalyzer(struct, symprec=0.25).get_crystal_system()
    except Exception:
        a, b, c, al, be, ga = struct.lattice.parameters
        ang = np.array([al, be, ga])
        if np.allclose(ang, 90.0, atol=0.5):
            if np.allclose([a, b, c], a, rtol=0.01):
                return "cubic"
            return "tetragonal" if np.isclose(a, b, rtol=0.01) else "orthorhombic"
        if np.isclose(ga, 120.0, atol=0.5) and np.isclose(a, b, rtol=0.01):
            return "hexagonal"
        if np.isclose(be, 90.0, atol=2.0) or np.isclose(al, 90.0, atol=2.0):
            return "monoclinic"
        return "triclinic"


def score_candidate(cand: Candidate, exp: ExpPattern, cfg: ScoreConfig, *,
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


def rank_candidates(candidates: Sequence[Candidate], exp: ExpPattern,
                    cfg: Optional[ScoreConfig] = None, *, top_k: int = 3,
                    build_times: Optional[Dict[str, float]] = None
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                               List[Dict[str, Any]]]:
    """按 Rwp 升序返回 (top_k, 全部成功, 全部失败)。"""
    cfg = cfg or ScoreConfig()
    build_times = build_times or {}
    scores = [score_candidate(c, exp, cfg,
                              build_time=build_times.get(f"{c.net_id}|{c.stacking}"))
              for c in candidates]
    ok = [s for s in scores if s["error"] is None]
    bad = [s for s in scores if s["error"] is not None]
    ok.sort(key=lambda s: (s["rwp"], s["geo_compliance"]))
    bad.sort(key=lambda s: s["net_id"])
    return (ok + bad)[:top_k], ok, bad


def diagnose(best: Dict[str, Any], exp: Optional[ExpPattern] = None) -> str:
    if not best:
        return "[diagnose] 无可用候选"
    L = [f"最佳候选: {best.get('net_id')}/{best.get('stacking')}"]
    if best.get("error"):
        L.append(f"  失败: {best['error']}")
        return "\n".join(L)
    m, p = best["metric"], best["profile"]
    L.append(f"  Rwp={best['rwp']*100:.2f}%   cos={best['cos_sim']:.4f}")
    L.append(f"  crystal_system={best.get('symmetry')}")
    L.append(f"  c={m['c']:.3f} Å  λa={m['lam_a']:.4f}  λb={m['lam_b']:.4f}"
             f"  zero={m['zero']:+.3f}°")
    L.append(f"  fwhm(面内)={p['fwhm_inplane']:.3f}° "
             f"fwhm(堆积)={p['fwhm_stack']:.3f}° "
             f"disp={p['disp']:+.3f}°  PO r={p['po_r']:.3f}")
    flag = "OK" if best["geo_compliance"] < 0.02 else "WARN(节点/连接子尺寸不匹配)"
    L.append(f"  几何合规度 |λ-1|max={best['geo_compliance']*100:.2f}% -> {flag}")
    if best.get("c_at_bound"):
        L.append("  ! c 撞到拟合边界 -> 001 峰信息不足，c 不可信")
    if exp is not None and exp.meta:
        L.append(f"  实验谱: {exp.meta.get('n_peaks')} 峰, "
                 f"低角遮蔽 <{exp.meta.get('low_angle_cut')}°")
    return "\n".join(L)
