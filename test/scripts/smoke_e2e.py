"""scripts/smoke_e2e.py — 端到端冒烟（v2.1）

修掉三处：
  * top_peaks 用 find_peaks 取真峰值（原版把同一峰的 6 个采样点当 6 个峰）
  * A（slab）用 slab_mode 跳过密度检查
  * 打印 a/b/c，便于判断节点是不是矩形
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.signal import find_peaks

from builder_bridge.pcb_adapter import (BBHandle, build_framework,
                                        framework_to_structure,
                                        load_building_block, validate_cif_structure)
from config.topology_registry import resolve_nets
from pxrd_engine.forward_scorer import (build_reflection_table, compute_pattern,
                                        symmetry_of, TWO_THETA_STEP)

PAIRS = [
    ("T3_BENZ_NH2_H", "NH2", "T3", "L2_BENZ_CHO_H", "CHO", "L2"),
]
DIAG_STACKINGS = ("A", "AA", "AAl", "AAt")


def top_peaks(grid, y, n=6):
    if y.size < 5 or float(np.max(y)) <= 0:
        return []
    pk, props = find_peaks(y, prominence=0.02 * float(np.max(y)),
                           distance=max(1, int(0.10 / TWO_THETA_STEP)))
    if pk.size == 0:
        return []
    order = np.argsort(props["prominences"])[::-1][:n]
    return sorted(round(float(grid[p]), 2) for p in pk[order])


def run_pair(name1, conn1, sym1, name2, conn2, sym2) -> bool:
    print("=" * 78)
    print(f"PAIR: {name1} ({sym1}/{conn1})  +  {name2} ({sym2}/{conn2})")
    print("=" * 78)
    try:
        bb1 = load_building_block(BBHandle(name1, conn1, "builtin", symmetry=sym1))
        bb2 = load_building_block(BBHandle(name2, conn2, "builtin", symmetry=sym2))
    except Exception:
        traceback.print_exc()
        return False

    nets = resolve_nets(sym1, sym2)
    if not nets:
        print("  !! 无可用 net")
        return False
    print(f"  nets: {[n.net_id for n in nets]}")

    all_ok = True
    store = {}
    for net in nets:
        for st in DIAG_STACKINGS:
            tag = f"{net.net_id}/{st}"
            try:
                fw = build_framework(bb1, bb2, net.net_token, st)
                s = framework_to_structure(fw, check_geometry=False)
                rep = validate_cif_structure(s, slab_mode=(st == "A"))
                table = build_reflection_table(s, tt_range=(1.5, 33.0))
                grid, y = compute_pattern(s, table)
                store[(net.net_id, st)] = (grid, y)
                a, b, c = s.lattice.abc
                print(f"  [{tag:13s}] {'PASS' if rep.ok else 'FAIL'} "
                      f"atoms={len(s):4d}  a={a:6.2f} b={b:6.2f} c={c:5.2f}  "
                      f"sg={rep.metrics.get('sg_loose')}  "
                      f"n_ref={table.hkl.shape[0]:4d} n00l={int(table.is_stack.sum()):3d}")
                print(f"                peaks={top_peaks(grid, y)}  "
                      f"d_min={rep.metrics.get('min_distance')}  "
                      f"sys={symmetry_of(s)}")
                if not rep.ok:
                    all_ok = False
                    for e in rep.errors:
                        print(f"                x {e}")
                for w in rep.warnings:
                    print(f"                ! {w}")
            except Exception as exc:
                all_ok = False
                print(f"  [{tag:13s}] EXC {type(exc).__name__}: {str(exc)[:110]}")

    for net in nets:
        ka, kb = (net.net_id, "A"), (net.net_id, "AA")
        if ka in store and kb in store:
            g, ya = store[ka]
            _, yaa = store[kb]
            only = sorted(set(top_peaks(g, yaa, 8)) - set(top_peaks(g, ya, 8)))
            print(f"  --- {net.net_id}: 仅 AA 出现的峰（≈00l）: {only}")
    return all_ok


def main() -> int:
    ok = True
    for p in PAIRS:
        try:
            ok &= run_pair(*p)
        except Exception:
            traceback.print_exc()
            ok = False
    print("\n" + "=" * 78)
    print("SMOKE:", "ALL PASS" if ok else "有失败项")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
