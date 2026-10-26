"""scripts/probe_bb_library.py

穷举 pcb 真实可用的构筑基元 -> config/bb_registry.json

为什么要穷举：0.0.9.4 里
  - get_available_core() 返回的是【对称性】(L2/T3/S4/D4/R4/H6)，不是 core 名；
  - get_buildingblock_list() 会返回不存在的 core（S4_TFPB / S4_PHEN）。
唯一可靠的办法：从 pcb 的报错信息里挖出真实 core 列表，再逐个真加载验证。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from builder_bridge.pcb_adapter import import_pcb

CORE_RE = re.compile(r"Available cores with (\w+) symmetry are \[(.*?)\]", re.S)
PATTERNS = ["", "_H", "_H_H", "_H_H_H", "_H_H_H_H", "_H_H_H_H_H",
            "_H_H_H_H_H_H", "_OH", "_H_OH", "_OH_H", "_OH_OH", "_OH_OH_OH"]
CONN_LIB = ("CHO", "NH2", "BOH2", "COOH", "OH2", "CONHNH2")


def capture(fn, *a, **kw) -> str:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fn(*a, **kw)
    except Exception as exc:
        buf.write(f"\n{type(exc).__name__}: {exc}")
    return buf.getvalue()


def cores_for_sym(pcb, sym: str) -> List[str]:
    txt = capture(pcb.BuildingBlock, name=f"{sym}_ZZZZ_ZZ", conector="NH2")
    m = CORE_RE.search(txt)
    if not m:
        print(f"  !! 无法从报错中解析 {sym} 的 core 列表")
        return []
    return [x.strip().strip("'\"") for x in m.group(2).split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="S4")
    ap.add_argument("--conn", default="CHO")
    ap.add_argument("--all", action="store_true", help="枚举全部 4 个对称性")
    args = ap.parse_args()

    pcb = import_pcb()
    probe = pcb.BuildingBlock()

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r_groups = list(probe.get_available_R() or [])
    except Exception:
        r_groups = []
    print("get_available_R():", r_groups)

    syms = ["L2", "T3", "S4", "H6"] if args.all else [args.sym]
    conns = list(CONN_LIB) if args.all else [args.conn]

    registry: Dict[str, List[Dict]] = {}
    tried = 0
    for sym in syms:
        cores = cores_for_sym(pcb, sym)
        print(f"\n{sym} cores ({len(cores)}): {cores}")
        for core in cores:
            for conn in conns:
                for suf in PATTERNS:
                    name = f"{sym}_{core}_{conn}{suf}"
                    tried += 1
                    try:
                        with contextlib.redirect_stdout(io.StringIO()), \
                             contextlib.redirect_stderr(io.StringIO()):
                            bb = pcb.BuildingBlock(name=name, conector=conn)
                        n_atoms = int(getattr(bb, "n_atoms", 0) or 0)
                        if n_atoms <= 0:
                            continue
                        try:
                            pts = bb.get_X_points()
                            n_x = int(len(pts[1])) if isinstance(pts, tuple) else int(len(pts))
                        except Exception:
                            n_x = -1
                        try:
                            mass = float(getattr(bb, "mass", 0.0) or 0.0)
                        except Exception:
                            mass = 0.0
                        registry.setdefault(sym, []).append(
                            dict(name=name, core=core, connector=conn,
                                 n_atoms=n_atoms, n_X=n_x, mass=round(mass, 2)))
                    except Exception:
                        continue

    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "config", "bb_registry.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in registry.values())
    print(f"\n尝试 {tried} 次，命中 {total} 个")
    for sym in syms:
        items = registry.get(sym, [])
        print(f"\n=== {sym} ({len(items)}) ===")
        for it in sorted(items, key=lambda d: (-d["n_X"], d["n_atoms"]))[:25]:
            print(f"  {it['name']:34s} atoms={it['n_atoms']:3d} "
                  f"X={it['n_X']:2d} mass={it['mass']:7.1f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
