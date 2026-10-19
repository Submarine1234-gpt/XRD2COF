"""COF 晶体粗粒化模块 —— 芳香环 → 带方向 CG 粒子。

策略：
  1. JmolNN 构建全单胞共价键图
  2. cycle_basis 找出所有最小环，筛选 5-7 元芳香环候选
  3. 每个芳香环 → 一个 CG 粒子：
     - 位置 = 环内原子的几何质心（分数坐标，PBC 展开后 wrap 回 [0,1)）
     - 方向 = 环平面法向量（SVD 拟合）
  4. 对称性自检：原结构 vs CG 假原子结构的空间群是否一致
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ═══════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CoarseGrainedStructure:
    """粗粒化后的结构表示。

    Attributes:
        pos:          [N, 3] float32  CG 粒子质心的分数坐标
        orientation:  [N, 3] float32  每个 CG 粒子的方向向量（环平面法向量，笛卡尔坐标）
        lattice:      [3, 3] float32  晶格矩阵 (Å)
        cg_type:      [N]   int64     环类型编码 (按环大小: 5→0, 6→1, 7→2)
        ring_sizes:   [N]   int64     每个 CG 粒子对应的环原子数
        atom_index:   [M]   int64     所有环内原子的原始 site 索引（扁平）
        atom_ptr:     [N+1] int64     CSR 分段指针
        space_group:  int or None     原结构空间群编号
        sg_cg:        int or None     CG 假原子结构空间群编号
        symmetry_ok:  bool            对称性自检是否通过
        n_rings:      int             识别的芳香环总数
        n_atoms:      int             原结构总原子数
        ring_elements: list[dict]     每个环的元素组成
        warnings:     list[str]       警告信息
    """
    pos: np.ndarray
    orientation: np.ndarray
    lattice: np.ndarray
    cg_type: np.ndarray
    ring_sizes: np.ndarray
    atom_index: np.ndarray
    atom_ptr: np.ndarray
    space_group: Optional[int] = None
    sg_cg: Optional[int] = None
    symmetry_ok: bool = True
    n_rings: int = 0
    n_atoms: int = 0
    ring_elements: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"CoarseGrainedStructure("
            f"n_rings={self.n_rings}, "
            f"n_atoms={self.n_atoms}, "
            f"pos={self.pos.shape}, "
            f"orientation={self.orientation.shape}, "
            f"SG_orig={self.space_group}, SG_cg={self.sg_cg}, "
            f"symmetry_ok={self.symmetry_ok}, "
            f"ring_sizes={self.ring_sizes.tolist()}, "
            f"warnings={self.warnings})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PBC 展开
# ═══════════════════════════════════════════════════════════════════════════

def _unwrap_ring(frac_coords: np.ndarray, ring_indices: list[int]) -> np.ndarray:
    """最小镜像展开一个环内的原子分数坐标。

    以环内第一个原子为锚点，将其余原子拉到与锚点同一镜像中，
    防止跨 PBC 边界的环质心计算错误（如 x=0.99 和 x=0.01 的平均不应是 0.50）。

    Returns:
        [len(ring_indices), 3] float64  展开后的分数坐标
    """
    ring_frac = frac_coords[ring_indices].copy()
    anchor = ring_frac[0]
    for i in range(1, len(ring_frac)):
        delta = ring_frac[i] - anchor
        ring_frac[i] -= np.round(delta)
    return ring_frac


# ═══════════════════════════════════════════════════════════════════════════
# 环法向量计算
# ═══════════════════════════════════════════════════════════════════════════

def _ring_normal(cart_coords: np.ndarray) -> np.ndarray:
    """计算一组（近）共面原子的平面法向量。

    使用 SVD 拟合平面：最小奇异值对应的右奇异向量即为法向量。
    法向量归一化为单位向量。

    Parameters:
        cart_coords: [K, 3] 笛卡尔坐标

    Returns:
        [3] float64 单位法向量
    """
    centered = cart_coords - np.mean(cart_coords, axis=0)
    _, s, vh = np.linalg.svd(centered)
    normal = vh[-1]  # 最小奇异值对应方向
    # 归一化（理论上 SVD 已归一化，但保险起见）
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        normal /= norm
    return normal


# ═══════════════════════════════════════════════════════════════════════════
# 环检测
# ═══════════════════════════════════════════════════════════════════════════

def _find_aromatic_rings(structure: Structure, graph: nx.Graph,
                         min_size: int = 5, max_size: int = 7
                         ) -> list[list[int]]:
    """从键图中找出所有芳香环候选（5-7 元环）。

    使用 networkx 的 cycle_basis 找最小环基，然后按大小和元素过滤。
    芳香环候选条件：
      - 环大小在 [min_size, max_size] 范围内
      - 环内原子全部是 C 或 N（典型芳香环元素）

    Returns:
        list of ring index lists, 每个 list 包含环内 site 索引
    """
    cycles = nx.cycle_basis(graph)
    aromatic_elements = {"C", "N", "B"}  # 常见芳香环元素

    rings = []
    for cycle in cycles:
        if min_size <= len(cycle) <= max_size:
            elems = {structure[i].specie.symbol for i in cycle}
            if elems.issubset(aromatic_elements):
                rings.append(sorted(cycle))

    return rings


# ═══════════════════════════════════════════════════════════════════════════
# 对称性自检
# ═══════════════════════════════════════════════════════════════════════════

def _check_symmetry(structure: Structure, cg_frac_pos: np.ndarray,
                    symprec: float = 0.01, angle_tolerance: float = 5.0,
                    dummy_element: str = "He") -> Tuple[bool, int, int, list]:
    """比较原结构与 CG 假原子结构的空间群编号。"""
    errors: list[str] = []
    sg_orig = sg_cg = None

    try:
        sg_orig = SpacegroupAnalyzer(
            structure, symprec=symprec, angle_tolerance=angle_tolerance
        ).get_space_group_number()
    except Exception as e:
        errors.append(f"原结构 SG 分析失败: {type(e).__name__}: {e}")

    try:
        n = cg_frac_pos.shape[0]
        cg_struct = Structure(
            structure.lattice,
            [dummy_element] * n,
            np.asarray(cg_frac_pos, dtype=np.float64),
        )
        sg_cg = SpacegroupAnalyzer(
            cg_struct, symprec=symprec, angle_tolerance=angle_tolerance
        ).get_space_group_number()
    except Exception as e:
        errors.append(f"CG 结构 SG 分析失败: {type(e).__name__}: {e}")

    ok = (sg_orig is not None and sg_cg is not None
          and sg_orig == sg_cg and not errors)
    return ok, sg_orig, sg_cg, errors


# ═══════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════

def coarse_grain_crystal(structure: Structure,
                         symprec: float = 0.01,
                         angle_tolerance: float = 5.0,
                         min_ring_size: int = 5,
                         max_ring_size: int = 7,
                         ) -> CoarseGrainedStructure:
    """对 COF 晶体执行芳香环粗粒化。

    流程：
      1. JmolNN → 共价键图
      2. cycle_basis → 筛选芳香环
      3. 每个环 → (质心, 法向量) 构成一个 CG 粒子
      4. 对称性自检

    Parameters:
        structure:       pymatgen Structure
        symprec:         SpacegroupAnalyzer 容差 (Å)
        angle_tolerance: 角度容差 (deg)
        min_ring_size:   最小环原子数（含）
        max_ring_size:   最大环原子数（含）

    Returns:
        CoarseGrainedStructure
    """
    warnings: list[str] = []

    # 1. 构建键图
    jmol = JmolNN()
    sg_graph = StructureGraph.from_local_env_strategy(structure, jmol)
    # MultiGraph → 简单 Graph（cycle_basis 不支持 MultiGraph）
    mg = sg_graph.graph.to_undirected()
    graph = nx.Graph(mg)

    # 2. 检测芳香环
    rings = _find_aromatic_rings(structure, graph, min_ring_size, max_ring_size)

    if len(rings) == 0:
        raise ValueError(
            f"未检测到 {min_ring_size}-{max_ring_size} 元芳香环，"
            "无法执行粗粒化。请检查 CIF 文件或调整 ring_size 参数。"
        )

    frac_coords = np.asarray(structure.frac_coords, dtype=np.float64)

    cg_positions = []   # 质心（分数坐标）
    cg_normals = []     # 法向量（笛卡尔坐标）
    cg_types = []       # 环类型编码
    ring_size_list = []
    ring_elem_list = []
    atom_indices_flat = []
    atom_ptrs = [0]

    # 环类型映射：环大小 → 类型编码
    type_map = {5: 0, 6: 1, 7: 2}

    for ring_indices in rings:
        # 3a. PBC 展开 + 几何质心
        unwrapped_frac = _unwrap_ring(frac_coords, ring_indices)
        centroid_frac = np.mean(unwrapped_frac, axis=0) % 1.0

        # 3b. 笛卡尔坐标下计算法向量
        cart_coords = structure.lattice.get_cartesian_coords(unwrapped_frac)
        normal = _ring_normal(cart_coords)

        # 元素组成
        from collections import Counter
        elems = Counter(structure[i].specie.symbol for i in ring_indices)

        cg_positions.append(centroid_frac)
        cg_normals.append(normal)
        cg_types.append(type_map.get(len(ring_indices), len(ring_indices)))
        ring_size_list.append(len(ring_indices))
        ring_elem_list.append(dict(elems))

        atom_indices_flat.extend(ring_indices)
        atom_ptrs.append(atom_ptrs[-1] + len(ring_indices))

    pos = np.array(cg_positions, dtype=np.float32)          # [N, 3]
    orientation = np.array(cg_normals, dtype=np.float32)     # [N, 3]
    cg_type = np.array(cg_types, dtype=np.int64)             # [N]
    ring_sizes = np.array(ring_size_list, dtype=np.int64)    # [N]
    atom_index = np.array(atom_indices_flat, dtype=np.int64) # [M]
    atom_ptr = np.array(atom_ptrs, dtype=np.int64)           # [N+1]

    # 4. 对称性自检（用 float64 避免精度问题）
    pos_f64 = np.array(cg_positions, dtype=np.float64)
    sym_ok, sg_orig, sg_cg, sym_errors = _check_symmetry(
        structure, pos_f64, symprec=symprec, angle_tolerance=angle_tolerance
    )
    warnings.extend(sym_errors)
    if not sym_ok:
        warnings.append(f"对称性自检失败：原 SG={sg_orig}，CG SG={sg_cg}")

    return CoarseGrainedStructure(
        pos=pos,
        orientation=orientation,
        lattice=np.array(structure.lattice.matrix, dtype=np.float32),
        cg_type=cg_type,
        ring_sizes=ring_sizes,
        atom_index=atom_index,
        atom_ptr=atom_ptr,
        space_group=sg_orig,
        sg_cg=sg_cg,
        symmetry_ok=sym_ok,
        n_rings=len(rings),
        n_atoms=len(structure),
        ring_elements=ring_elem_list,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 命令行测试
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    cif_path = sys.argv[1] if len(sys.argv) > 1 else "Data/CoreCOF/2.cif"
    print(f"读取: {cif_path}")
    struct = Structure.from_file(cif_path)
    print(f"  sites = {len(struct)}, lattice = {struct.lattice.abc}")

    result = coarse_grain_crystal(struct, symprec=0.01)

    print(f"\n===== 粗粒化结果 =====")
    print(f"  芳香环数 (CG 粒子数): {result.n_rings}")
    print(f"  原子总数:             {result.n_atoms}")
    print(f"  压缩比:               {result.n_atoms / result.n_rings:.1f}:1")
    print(f"  空间群 (原/CG):       {result.space_group} / {result.sg_cg}")
    print(f"  对称性自检:           {'✓ 通过' if result.symmetry_ok else '✗ 失败'}")
    print(f"  环大小分布:           {result.ring_sizes.tolist()}")
    print()
    print(f"  pos (质心, 分数坐标):")
    for i, p in enumerate(result.pos):
        elem = result.ring_elements[i]
        print(f"    CG[{i}]: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]  "
              f"ring_size={result.ring_sizes[i]}  elements={elem}")
    print()
    print(f"  orientation (法向量, 笛卡尔):")
    for i, n in enumerate(result.orientation):
        print(f"    CG[{i}]: [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]")

    if result.warnings:
        print(f"\n  ⚠ 警告: {result.warnings}")