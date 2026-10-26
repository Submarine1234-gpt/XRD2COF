"""scripts/dump_structure.py

判定结构几何到底哪里坏了，三条对比：
  1. from_building_blocks 路径  vs  from_name 路径（同一 COF 名）
  2. to_ase() 的坐标  vs  cof.atom_pos
  3. 带 -OH 的节点  vs  不带 -OH 的同构节点

输出：元素计数、晶胞、10 个最短接触（含分数坐标）、按元素对的距离统计。
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from builder_bridge.pcb_adapter import (BBHandle, build_framework, import_pcb,
                                        framework_to_structure,
                                        load_building_block, quiet)


def dump(struct, tag: str) -> None:
    syms = [str(s.symbol) for s in struct.species]
    print(f"\n--- {tag} ---")
    print("  n_atoms:", len(struct), "| formula:", struct.composition.reduced_formula)
    print("  cell   :", [round(x, 3) for x in struct.lattice.abc],
          [round(x, 2) for x in struct.lattice.angles])

    dm = struct.distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    print("  10 最短接触:")
    flat = np.argsort(dm, axis=None)
    seen = set()
    shown = 0
    for k in flat:
        i, j = divmod(int(k), dm.shape[0])
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        print(f"    {syms[i]}{i:3d}-{syms[j]}{j:3d}  {dm[i, j]:7.3f} Å  "
              f"frac {np.round(struct.frac_coords[i], 3)} | "
              f"{np.round(struct.frac_coords[j], 3)}")
        shown += 1
        if shown >= 10:
            break

    agg = defaultdict(list)
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            if dm[i, j] < 2.4:
                agg[tuple(sorted((syms[i], syms[j])))].append(float(dm[i, j]))
    print("  元素对距离统计 (<2.4 Å):")
    for key in sorted(agg):
        v = np.array(agg[key])
        print(f"    {key[0]}-{key[1]}: n={len(v):4d}  min={v.min():.3f}  "
              f"med={np.median(v):.3f}  max={v.max():.3f}")


def main() -> int:
    pcb = import_pcb()

    # ---------- 1) from_building_blocks 路径 ----------
    bb1 = load_building_block(BBHandle("T3_BENZ_NH2_OH", "NH2", symmetry="T3"))
    bb2 = load_building_block(BBHandle("L2_BENZ_CHO_H", "CHO", symmetry="L2"))
    fw_a = build_framework(bb1, bb2, "HCB_A", "AA")
    s_a = framework_to_structure(fw_a, check_geometry=False)
    dump(s_a, "from_building_blocks: T3_BENZ_NH2_OH + L2_BENZ_CHO_H / HCB_A-AA")

    # ---------- 2) from_name 路径（官方示例） ----------
    try:
        with quiet():
            fw_b = pcb.Framework("T3_BENZ_NH2_OH-L2_BENZ_CHO_H-HCB_A-AA",
                                 verbosity="none")
        s_b = framework_to_structure(fw_b, check_geometry=False)
        dump(s_b, "from_name: 同一 COF 名")
        print("\n  >>> 两路径一致性: n_atoms "
              f"{len(s_a)} vs {len(s_b)} | "
              f"min_dist {s_a.distance_matrix[s_a.distance_matrix > 0].min():.3f} "
              f"vs {s_b.distance_matrix[s_b.distance_matrix > 0].min():.3f}")
    except Exception as exc:
        print(f"\n  from_name 路径失败: {type(exc).__name__}: {exc}")

    # ---------- 3) to_ase() 是否忠实 ----------
    print("\n--- to_ase() 自洽性 ---")
    at = fw_a.to_ase()
    print("  ase n_atoms:", len(at))
    print("  ase positions[:3]:\n", np.round(at.positions[:3], 3))
    if hasattr(fw_a, "atom_pos"):
        ap = np.asarray(fw_a.atom_pos, dtype=float)
        print("  cof.atom_pos[:3]:\n", np.round(ap[:3], 3), "shape", ap.shape)
        if ap.shape == at.positions.shape:
            print("  max|ase - atom_pos| =", float(np.abs(at.positions - ap).max()))
        else:
            print("  !! 形状不一致，to_ase 可能重排或丢了原子")
    else:
        print("  cof 无 atom_pos 属性")
    if hasattr(fw_a, "atom_types"):
        print("  cof.atom_types[:6]:", list(fw_a.atom_types)[:6])

    # ---------- 4) 去掉 -OH 的同构节点 ----------
    try:
        bb1h = load_building_block(BBHandle("T3_BENZ_NH2_H", "NH2", symmetry="T3"))
        fw_c = build_framework(bb1h, bb2, "HCB_A", "AA")
        dump(framework_to_structure(fw_c, check_geometry=False),
             "对照: T3_BENZ_NH2_H (无 -OH) + L2_BENZ_CHO_H")
    except Exception as exc:
        print(f"\n  对照组失败: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
