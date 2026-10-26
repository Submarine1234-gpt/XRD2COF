"""scripts/find_working_s4.py

找出 S4 + L2 组合下哪些 (基元, net, stacking) 真能建出几何自洽的结构。

要回答两个问题：
  Q1  S4_BENZ_CHO（矩形节点）在 SQL_A/AA 崩，是因为它是矩形吗？
      -> 用 S4_PORP_CHO（卟啉，D4h 真方形）做对照
  Q2  FXT_A 是不是上游全坏（issue #86）？
      -> 换任何基元都不行，则为上游 bug
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from builder_bridge.pcb_adapter import (BBHandle, build_framework,
                                        framework_to_structure, import_pcb,
                                        load_building_block, quiet,
                                        validate_cif_structure)

BASES_S4 = ["S4_BENZ_CHO", "S4_PORP_CHO", "S4_PTCA_CHO",
            "S4_OTPR_CHO", "S4_TBPR_CHO", "S4_PHPR_CHO"]
L2_NAME = "L2_BENZ_NH2_H"
NETS = ["SQL_A", "FXT_A"]
STACKS = ["A", "AA"]


def main() -> int:
    pcb = import_pcb()

    print("=== 基元可用性 ===")
    bases = []
    for n in BASES_S4:
        try:
            with quiet():
                bb = pcb.BuildingBlock(name=n, conector="CHO")
            n_atoms = int(getattr(bb, "n_atoms", 0) or 0)
            pts = bb.get_X_points()
            n_x = int(len(pts[1])) if isinstance(pts, tuple) else int(len(pts))
            print(f"  {n:16s} atoms={n_atoms:3d} X={n_x}  OK")
            if n_atoms > 0 and n_x == 4:
                bases.append(n)
        except Exception as exc:
            print(f"  {n:16s} FAIL: {type(exc).__name__}")

    bb_l2 = load_building_block(BBHandle(L2_NAME, "NH2", symmetry="L2"))
    print(f"\nL2: {L2_NAME} atoms={bb_l2.n_atoms}")

    print("\n=== 建胞矩阵 ===")
    print(f"{'base':16s} {'net':6s} {'stack':5s} {'结果':6s} "
          f"{'atoms':>5s} {'a':>7s} {'b':>7s} {'c':>6s} {'d_min':>6s}")
    print("-" * 78)
    winners: dict[str, list[str]] = {}

    for base in bases:
        try:
            bb1 = load_building_block(BBHandle(base, "CHO", symmetry="S4"))
        except Exception:
            continue
        for net in NETS:
            for st in STACKS:
                tag = f"{base}/{net}/{st}"
                try:
                    fw = build_framework(bb1, bb_l2, net, st)
                    struct = framework_to_structure(fw, check_geometry=False)
                    a, b, c = struct.lattice.abc
                    dm = struct.distance_matrix
                    np.fill_diagonal(dm, np.inf)
                    dmin = float(np.min(dm))
                    ok = "PASS"
                    if dmin < 0.9:
                        ok = "TOOCLOS"
                    else:
                        try:
                            rep = validate_cif_structure(struct, slab_mode=(st == "A"))
                            if not rep.ok:
                                ok = "INVALID"
                        except TypeError as te:
                            ok = "TOOLBUG"      # 校验器签名错，不是几何错
                            print("      !! validate 调用失败:", te)
                    print(f"{base:16s} {net:6s} {st:5s} {ok:6s} "
                          f"{len(struct):5d} {a:7.2f} {b:7.2f} {c:6.2f} {dmin:6.3f}")
                    if ok == "PASS":
                        winners.setdefault(base, []).append(f"{net}/{st}")
                except Exception as exc:
                    msg = str(exc)[:46].replace("\n", " ")
                    print(f"{base:16s} {net:6s} {st:5s} FAIL   {msg}")

    print("\n=== 可用组合 ===")
    if not winners:
        print("  （无）")
    for k, v in winners.items():
        print(f"  {k}: {v}")

    print("\n=== 判读 ===")
    benz_sql_aa = any(x.startswith("SQL_A/AA") for x in winners.get("S4_BENZ_CHO", []))
    porp_sql_aa = any(x.startswith("SQL_A/AA") for x in winners.get("S4_PORP_CHO", []))
    any_fxt = any(n.startswith("FXT_A") for v in winners.values() for n in v)

    if not benz_sql_aa and porp_sql_aa:
        print("  -> 假设1成立：S4_BENZ_CHO 是矩形节点，不能进 SQL_A。"
              "改用卟啉类真方形节点。")
    elif not benz_sql_aa and not porp_sql_aa:
        print("  -> 假设1不成立：SQL_A 的多层路径本身有问题（任何基元都崩）。")
    else:
        print("  -> S4_BENZ_CHO 在 SQL_A/AA 竟然可用，与之前结果矛盾，需复查。")

    if not any_fxt:
        print("  -> 假设2成立：FXT_A 所有基元均失败，确认为上游 bug (#86)，永久移除。")
    else:
        print("  -> FXT_A 部分可用，需逐个检查几何质量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
