"""scripts/probe_sql_aa.py — SQL_A 多层失败是否依赖 c_parameter_base / slab。"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from builder_bridge.pcb_adapter import (BBHandle, build_framework, import_pcb,
                                        load_building_block, quiet)

BASE = sys.argv[1] if len(sys.argv) > 1 else "S4_PORP_CHO"
C_GRID = [3.3, 3.6, 3.9, 4.2, 4.5, 5.0]
SLAB_GRID = [5.0, 10.0, 20.0]

def main() -> int:
    pcb = import_pcb()
    bb1 = load_building_block(BBHandle(BASE, "CHO", symmetry="S4"))
    bb2 = load_building_block(BBHandle("L2_BENZ_NH2_H", "NH2", symmetry="L2"))
    print(f"base={BASE}  atoms={bb1.n_atoms}\n")
    print(f"{'c_par':>6s} {'slab':>6s}  {'结果':10s} {'atoms':>5s} {'c':>6s} {'d_min':>7s}")
    print("-" * 52)

    for c in C_GRID:
        for slab in SLAB_GRID:
            try:
                fw = build_framework(bb1, bb2, "SQL_A", "AA",
                                     c_parameter_base=c, verbosity="none")
                # slab 不是 build_framework 的参数，直接走 pcb 原生调用
            except Exception:
                # 带 slab 再试一次
                try:
                    with quiet():
                        fw = pcb.Framework(verbosity="none")
                        fw.from_building_blocks(bb1, bb2, "SQL_A", "AA",
                                                c_parameter_base=c, slab=slab)
                except Exception as exc:
                    print(f"{c:6.1f} {slab:6.1f}  FAIL       {str(exc)[:70]}")
                    continue
            from builder_bridge.pcb_adapter import framework_to_structure
            st = framework_to_structure(fw, check_geometry=False)
            dm = st.distance_matrix
            np.fill_diagonal(dm, np.inf)
            dmin = float(np.min(dm))
            flag = "OK" if dmin > 0.9 else "TOOCLOS"
            print(f"{c:6.1f} {slab:6.1f}  {flag:10s} {len(st):5d} "
                  f"{st.lattice.c:6.2f} {dmin:7.3f}")
            break
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
