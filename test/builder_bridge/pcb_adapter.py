"""builder_bridge/pcb_adapter.py  —  pyCOFBuilder 适配层（0.0.9.x 加固）"""
from __future__ import annotations

import contextlib
import inspect
import io
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

try:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    _HAS_PMG_SYM = True
except Exception:
    _HAS_PMG_SYM = False

try:
    import spglib  # noqa: F401
    _HAS_SPGLIB = True
except Exception:
    _HAS_SPGLIB = False


def import_pcb():
    try:
        import pycofbuilder as pcb
    except Exception as exc:
        raise ImportError(
            "未找到 pycofbuilder。请先:\n"
            "  conda activate XRD2COF\n"
            "  pip install gemmi simplejson ase\n"
            "  pip install pycofbuilder"
        ) from exc
    return pcb


def _accepted_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


@contextlib.contextmanager
def quiet():
    """吞掉 pcb 打在 stdout/stderr 上的 "ERROR!: XXX not available!"。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _try_call(fn, *args, **kwargs):
    try:
        with quiet():
            return True, fn(*args, **_accepted_kwargs(fn, kwargs)), None
    except Exception as exc:
        return False, None, exc


@dataclass
class BBHandle:
    """一个构筑基元的引用（内置名 或 自定义 .xyz 文件）。"""
    name: str
    connector: Optional[str] = None
    source: str = "builtin"
    path: Optional[str] = None
    symmetry: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source == "file" and not self.path:
            raise ValueError(f"{self.name}: source='file' 必须给 path")
        if self.symmetry is None and "_" in self.name:
            head = self.name.split("_")[0].upper()
            if head in {"L2", "T3", "S4", "D4", "H6"}:
                self.symmetry = head


def _extract_x_points(bb: Any) -> Optional[np.ndarray]:
    """get_X_points() 可能返回 (labels, points)，也可能只返回 points。"""
    fn = getattr(bb, "get_X_points", None)
    if fn is None:
        return None
    try:
        got = fn()
    except Exception:
        return None
    if isinstance(got, tuple):
        for item in got:
            try:
                arr = np.asarray(item, dtype=float)
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
        return None
    try:
        arr = np.asarray(got, dtype=float)
        return arr if arr.ndim == 2 and arr.shape[1] == 3 else None
    except Exception:
        return None


def _bb_meta(bb: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    pts = _extract_x_points(bb)
    meta["n_connections"] = int(len(pts)) if pts is not None else None
    for attr in ("smiles", "mass", "molecular_weight", "name"):
        if hasattr(bb, attr):
            try:
                val = getattr(bb, attr)
                if val not in (None, ""):
                    meta[attr] = val
            except Exception:
                pass
    if "mass" not in meta:
        try:
            from pymatgen.core import Composition
            types = list(getattr(bb, "atom_types", []) or [])
            if types:
                meta["mass"] = float(Composition({str(t): 1 for t in types}).weight)
        except Exception:
            pass
    meta.setdefault("mass", 0.0)
    return meta


def load_building_block(handle: BBHandle, *, out_dir: Optional[str] = None,
                        verbosity: str = "none") -> Any:
    pcb = import_pcb()
    tmp = out_dir or tempfile.mkdtemp(prefix="pcb_bb_")
    os.makedirs(tmp, exist_ok=True)
    common = dict(conector=handle.connector, connector=handle.connector,
                  out_dir=tmp, bb_dir=tmp, verbosity=verbosity)

    if handle.source == "file":
        base = os.path.basename(handle.path or "")
        folder = os.path.dirname(os.path.abspath(handle.path or ""))
        stem = os.path.splitext(base)[0]
        last = None
        for kw, args in [(dict(common, name=base, out_dir=folder, bb_dir=folder), ()),
                         (dict(common, name=base), ()),
                         (dict(common, file_name=base), ()),
                         (dict(common), (folder, base))]:
            ok, bb, err = _try_call(pcb.BuildingBlock, *args, **kw)
            if ok and bb is not None and getattr(bb, "n_atoms", 0) > 0:
                handle.meta.update(_bb_meta(bb))
                return bb
            last = err or last
        ok, bb, err2 = _try_call(pcb.BuildingBlock, **common)
        if ok and bb is not None and hasattr(bb, "from_file"):
            for a in ((folder, stem), (folder, base)):
                ok2, _, err3 = _try_call(bb.from_file, *a)
                if ok2 and getattr(bb, "n_atoms", 0) > 0:
                    handle.meta.update(_bb_meta(bb))
                    return bb
                last = err3 or last
        raise RuntimeError(f"无法从文件加载基元 {handle.path}: {err2 or last}")

    last = None
    for kw in [dict(common, name=handle.name),
               dict(common, name=handle.name, symmetry=handle.symmetry),
               dict(common)]:
        ok, bb, err = _try_call(pcb.BuildingBlock, **kw)
        if ok and bb is not None and getattr(bb, "n_atoms", 0) > 0:
            handle.meta.update(_bb_meta(bb))
            return bb
        last = err or last
        if ok and bb is not None and hasattr(bb, "from_name"):
            ok2, _, err2 = _try_call(bb.from_name, handle.name)
            if ok2 and getattr(bb, "n_atoms", 0) > 0:
                handle.meta.update(_bb_meta(bb))
                return bb
            last = err2 or last
    raise RuntimeError(f"无法加载内置基元 {handle.name!r}: {last}")


def new_framework(**kwargs) -> Any:
    pcb = import_pcb()
    tmp = kwargs.pop("out_dir", None) or tempfile.mkdtemp(prefix="pcb_fw_")
    base = dict(save_bb=False, out_dir=tmp)
    base.update(kwargs)                 # 调用方显式给的优先
    base.setdefault("verbosity", "none")
    ok, fw, _ = _try_call(pcb.Framework, **base)
    if ok and fw is not None:
        return fw
    ok, fw, _ = _try_call(pcb.Framework)
    if ok and fw is not None:
        return fw
    raise RuntimeError("pcb.Framework 实例化失败")


#: 只有这两个 net 的 create_* 函数接受 c_parameter_base（0.0.9.4 实测）
_C_PARAM_NETS = frozenset({"SQL_A", "FXT_A"})


def _available_tokens(fw: Any) -> set:
    """pcb 真实接受的 net token 集合，用于回退时过滤。"""
    try:
        toks = getattr(fw, "available_topologies", None)
        if toks:
            return {str(t) for t in toks}
    except Exception:
        pass
    return set()


def _is_bond_error(exc: Optional[BaseException]) -> bool:
    if exc is None:
        return False
    name = type(exc).__name__
    return "BondLengh" in name or "BondLength" in name


def build_framework(bb1: Any, bb2: Any, net_token: str, stacking: str,
                    *, shift_vector: Optional[Sequence[float]] = None,
                    tilt_angle: Optional[float] = None,
                    c_parameter_base: Optional[float] = None,
                    verbosity: str = "none") -> Any:
    """调用 from_building_blocks(bb1, bb2, net, stacking)。

    回退策略（0.0.9.4 实测校正）：
      - 只在 pcb 自己声明的 available_topologies 里回退，绝不猜小写；
      - 保留**第一个**错误，不随后续失败覆盖；
      - BondLenghError 单独语义化：表示该 (基元, net) 几何不兼容，
        不是 API 用法错误，重试无意义。
    """
    fw = new_framework(verbosity=verbosity)
    kwargs: Dict[str, Any] = {}
    if shift_vector is not None:
        kwargs["shift_vector"] = list(shift_vector)
    if tilt_angle is not None:
        kwargs["tilt_angle"] = float(tilt_angle)
    if c_parameter_base is not None and net_token.upper() in _C_PARAM_NETS:
        kwargs["c_parameter_base"] = float(c_parameter_base)

    ok, _, err = _try_call(fw.from_building_blocks, bb1, bb2, net_token,
                           stacking, **kwargs)
    if ok:
        if getattr(fw, "n_atoms", 0) <= 0:
            raise RuntimeError(f"{net_token}/{stacking}: 生成的框架原子数为 0")
        return fw

    first_err = err
    if _is_bond_error(first_err):
        raise RuntimeError(
            f"{net_token}/{stacking} 几何不兼容（上游 BondLenghError: {first_err}）。"
            "这不是 API 用法问题——该基元与该 net 无法拼接，重试无意义。")

    # 仅在 pcb 声明的 token 集合内尝试别名
    allowed = _available_tokens(fw)
    alts = [a for a in (net_token.upper(), net_token.replace("_A", ""))
            if a != net_token and (not allowed or a in allowed)]
    for alt in alts:
        ok, _, err2 = _try_call(fw.from_building_blocks, bb1, bb2, alt,
                                stacking, **kwargs)
        if ok and getattr(fw, "n_atoms", 0) > 0:
            return fw

    raise RuntimeError(
        f"from_building_blocks({net_token}, {stacking}) 失败: {first_err}"
        + (f"  [已尝试别名 {alts}]" if alts else ""))


def framework_to_structure(fw: Any, *, check_geometry: bool = True) -> Structure:
    """Framework -> Structure，并做几何自检（拦坐标系数值错用）。"""
    struct = None
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        struct = AseAtomsAdaptor.get_structure(fw.to_ase())
    except Exception:
        struct = None

    if struct is None:
        lat = np.asarray(getattr(fw, "lattice"), dtype=float)
        types = list(getattr(fw, "atom_types"))
        pos = np.asarray(getattr(fw, "atom_pos"), dtype=float)
        if lat.shape != (3, 3) or len(pos) != len(types) or not types:
            raise RuntimeError(
                f"Framework 数组形状异常 lat={lat.shape} pos={pos.shape} "
                f"types={len(types)}")
        struct = Structure(Lattice(lat), types, pos, coords_are_cartesian=True)

    if check_geometry:
        dm = struct.distance_matrix
        np.fill_diagonal(dm, np.inf)
        dmin = float(np.min(dm)) if dm.size else np.inf
        if dmin < 0.9:
            raise RuntimeError(
                f"最近邻距离仅 {dmin:.3f} Å —— 坐标系假定可能有误。"
                "若 pcb 的 atom_pos 是分数坐标，请把 "
                "coords_are_cartesian 改为 False。")
    return struct


def set_c_parameter(struct: Structure, c_target: float) -> Structure:
    """只缩放 c 晶格矢量，分数坐标严格不动（2D 层的精确变换）。"""
    mat = struct.lattice.matrix.copy()
    c0 = float(np.linalg.norm(mat[2]))
    if c0 <= 1e-6:
        raise ValueError("c 轴长度为 0，晶格异常")
    mat[2] *= float(c_target) / c0
    return Structure(Lattice(mat), struct.species, struct.frac_coords)


def apply_inplane_scale(struct: Structure, lam_a: float, lam_b: float,
                        *, warn_above: float = 0.02) -> Structure:
    """仅诊断用。各向异性缩放 a/b 会剪切层内几何，破坏键长。"""
    if abs(lam_a - lam_b) > warn_above:
        warnings.warn(
            f"|λa-λb|={abs(lam_a - lam_b):.3f} > {warn_above}: 层内几何被剪切，"
            "键长不再可信，该结构不得作为结果输出。",
            RuntimeWarning, stacklevel=2)
    mat = struct.lattice.matrix.copy()
    mat[0] *= lam_a
    mat[1] *= lam_b
    return Structure(Lattice(mat), struct.species, struct.frac_coords)


@dataclass
class ValidationReport:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise AssertionError("结构校验失败:\n  - " + "\n  - ".join(self.errors))

    def summary(self) -> str:
        lines = [f"[validate] {'PASS' if self.ok else 'FAIL'}  metrics={self.metrics}"]
        lines += [f"  ! {w}" for w in self.warnings]
        lines += [f"  x {e}" for e in self.errors]
        return "\n".join(lines)


_COVALENT_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}


def validate_cif_structure(struct: Structure, *,
                           min_distance: float = 0.90,
                           bond_ratio: Tuple[float, float] = (0.70, 1.35),
                           strict_symmetry: bool = False,
                           symprec_loose: float = 0.25,
                           symprec_tight: float = 0.05,
                           slab_mode: bool = False) -> ValidationReport:
    """落盘前的硬断言网关。

    改进点：dist/(r_i+r_j) > bond_ratio[1] 的**不算错误**——2.0~2.2 Å 的
    H···C / H···H 是正常范德华接触，不是坏键。只有「太短」才是结构坏。
    单独保留 H-H < 1.20 Å 的硬检查（真正的原子重叠信号）。
    """
    rep = ValidationReport(ok=True)
    syms = [str(s.symbol) for s in struct.species]

    dm = struct.distance_matrix
    np.fill_diagonal(dm, np.inf)
    dmin = float(np.min(dm)) if dm.size else np.inf
    rep.metrics["min_distance"] = round(dmin, 4)
    if dmin < min_distance:
        i, j = np.unravel_index(np.argmin(dm), dm.shape)
        rep.errors.append(
            f"最近邻距离 {dmin:.3f} Å < {min_distance} Å ({syms[i]}{i}-{syms[j]}{j})")

    too_short: List[str] = []
    hh_bad: List[str] = []
    n_contact = 0
    n = len(syms)
    if n <= 800:
        for i in range(n):
            for j in range(i + 1, n):
                d = float(dm[i, j])
                if d > 2.2:
                    continue
                si, sj = syms[i], syms[j]
                if si == "H" and sj == "H" and d < 1.20:
                    hh_bad.append(f"H{i}-H{j}:{d:.2f}Å")
                ri, rj = _COVALENT_RADII.get(si), _COVALENT_RADII.get(sj)
                if not ri or not rj:
                    continue
                ratio = d / (ri + rj)
                if ratio < bond_ratio[0]:
                    too_short.append(f"{si}{i}-{sj}{j}:{d:.2f}(r={ratio:.2f})")
                else:
                    n_contact += 1
    else:
        rep.warnings.append(f"原子数 {n} > 800，跳过逐键长校验")

    if too_short:
        rep.errors.append(
            f"{len(too_short)} 处原子过近（键长比 < {bond_ratio[0]}），"
            f"前5: {too_short[:5]}")
    if hh_bad:
        rep.errors.append(f"{len(hh_bad)} 处 H-H < 1.20 Å（不可能的几何），"
                          f"前5: {hh_bad[:5]}")
    rep.metrics["n_too_short"] = len(too_short)
    rep.metrics["n_HH_too_close"] = len(hh_bad)
    rep.metrics["n_vdw_contacts"] = n_contact

    if "X" in {s.split()[0] for s in syms}:
        rep.errors.append("结构中仍含 X 虚原子（连接点未被替换）")

    vpa = struct.volume / max(n, 1)
    rep.metrics["volume_per_atom"] = round(float(vpa), 2)
    if slab_mode:
        rep.metrics["slab_mode"] = True
        rep.warnings.append("slab 模式（单层 + 真空层），跳过密度检查")
    elif not (5.0 <= vpa <= 60.0):
        rep.errors.append(f"每原子体积 {vpa:.1f} Å³ 超出 [5, 60]")

    if _HAS_SPGLIB and _HAS_PMG_SYM:
        try:
            lo = SpacegroupAnalyzer(struct, symprec=symprec_loose)
            ti = SpacegroupAnalyzer(struct, symprec=symprec_tight)
            nl, nt = lo.get_space_group_number(), ti.get_space_group_number()
            rep.metrics["sg_loose"] = (nl, lo.get_space_group_symbol())
            rep.metrics["sg_tight"] = (nt, ti.get_space_group_symbol())
            rep.metrics["crystal_system"] = lo.get_crystal_system()
            if nl != nt:
                msg = (f"空间群随 symprec 漂移: {nt}@{symprec_tight} -> "
                       f"{nl}@{symprec_loose}")
                (rep.errors if strict_symmetry else rep.warnings).append(msg)
        except Exception as exc:
            rep.warnings.append(f"spglib 分析异常: {exc}")
    else:
        rep.warnings.append("spglib 不可用，跳过空间群校验")

    rep.ok = not rep.errors
    return rep


def write_cif(struct: Structure, path: str, *, symprec: float = 0.25,
              validate: bool = True, strict_symmetry: bool = False) -> ValidationReport:
    rep = ValidationReport(ok=True)
    if validate:
        rep = validate_cif_structure(struct, strict_symmetry=strict_symmetry)
        rep.raise_if_failed()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    CifWriter(struct, symprec=symprec).write_file(path)
    return rep


def describe_pcb_api() -> str:
    try:
        pcb = import_pcb()
    except ImportError as exc:
        return str(exc)
    lines = [f"pycofbuilder: {getattr(pcb, '__version__', 'unknown')}"]
    try:
        fw = new_framework()
    except Exception as exc:
        lines.append(f"  !! Framework 实例化失败: {exc}")
        return "\n".join(lines)

    for name in ("from_building_blocks", "from_name",
                 "create_hcb_a_structure", "create_hcb_structure",
                 "create_sql_a_structure", "create_sql_structure",
                 "create_fxt_a_structure", "create_fxt_structure",
                 "create_kgd_structure", "create_hxl_a_structure",
                 "save", "to_ase"):
        fn = getattr(fw, name, None)
        if fn is None:
            lines.append(f"  {name}: MISSING")
            continue
        try:
            lines.append(f"  {name}: {inspect.signature(fn)}")
        except (TypeError, ValueError):
            lines.append(f"  {name}: <signature unavailable>")

    for attr in ("available_2D_topologies", "available_3D_topologies",
                 "available_topologies", "available_stacking",
                 "symm_tol", "angle_tol", "dist_threshold"):
        if hasattr(fw, attr):
            try:
                lines.append(f"  {attr} = {getattr(fw, attr)}")
            except Exception:
                pass

    bb_cls = getattr(pcb, "BuildingBlock", None)
    if bb_cls is not None:
        lines.append("  --- BuildingBlock ---")
        for name in ("from_file", "from_name", "get_X_points", "save",
                     "to_ase", "get_buildingblock_list", "replace_X"):
            fn = getattr(bb_cls, name, None)
            if fn is None:
                lines.append(f"    {name}: MISSING")
                continue
            try:
                lines.append(f"    {name}: {inspect.signature(fn)}")
            except (TypeError, ValueError):
                lines.append(f"    {name}: <signature unavailable>")
    return "\n".join(lines)
