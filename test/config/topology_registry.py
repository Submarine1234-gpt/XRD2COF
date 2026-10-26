"""
config/topology_registry.py

与 pyCOFBuilder 实际实现的 2D 建网方法严格对齐的拓扑注册表。

对齐依据（pcb 0.0.8.x）：
    create_hcb_a_structure(BB_T3, BB_L2, stacking, slab, shift_vector, tilt_angle)
    create_hcb_structure(BB_T3, BB_T3, stacking, ...)
    create_sql_a_structure(BB_S4, BB_L2, stacking, c_parameter_base, slab, shift_vector, tilt_angle)
    create_sql_structure(BB_S4, BB_S4, stacking, slab, shift_vector, tilt_angle)
    create_fxt_a_structure(BB_S4, BB_L2, stacking, c_parameter_base, slab, shift_vector, tilt_angle)
    create_fxt_structure(BB_R4_A, BB_R4_B, stacking, slab, shift_vector, tilt_angle)
    create_kgd_structure(BB_H6, BB_T3, stacking, slab, shift_vector, tilt_angle)
    create_hxl_a_structure(BB_H6, BB_L2, stacking, slab, shift_vector, tilt_angle)

任何不在此表内的 (对称性, 对称性) 组合都必须显式返回「未实现」，
绝不允许静默回退到某个近似拓扑。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------
# 基元对称性类别（pcb 命名法的第一个字段）
# ----------------------------------------------------------------------------
BB_SYMMETRIES: Tuple[str, ...] = ("L2", "T3", "S4", "D4", "H6")

#: 名义连接数。D4 是 3D 四面体节点，与 2D 建网无关。
CONNECTIVITY: Dict[str, int] = {"L2": 2, "T3": 3, "S4": 4, "D4": 4, "H6": 6}

#: 2D 堆积模式词表。AA=全重叠；AAl=层间滑移(slipped)；AAt=层间倾斜(tilted)。
#: AB / ABC 在部分版本里存在，但物理意义弱且实验判别力低，默认关闭。
STACKING_2D_BASE: Tuple[str, ...] = ("AA", "AAl", "AAt")
STACKING_2D_OPTIONAL: Tuple[str, ...] = ("AB", "ABC")


@dataclass(frozen=True)
class NetSpec:
    """一条「(BB1 对称性, BB2 对称性) -> 拓扑」的可执行记录。"""

    net_id: str          # 内部短名，如 "hcb_a"
    net_token: str       # pcb 传入名，如 "HCB_A"（注意 _A 后缀不能丢）
    builder: str         # pcb Framework 上的方法名
    bb1_sym: str         # 高连接度基元（pcb 命名法要求 BB1 连接数 >= BB2）
    bb2_sym: str
    stackings: Tuple[str, ...] = STACKING_2D_BASE
    default_stacking: str = "AA"
    enabled: bool = True
    note: str = ""

    @property
    def n_bb1(self) -> int:
        return CONNECTIVITY[self.bb1_sym]

    @property
    def n_bb2(self) -> int:
        return CONNECTIVITY[self.bb2_sym]

    @property
    def is_2d(self) -> bool:
        return "_3d" not in self.net_id

    def __post_init__(self) -> None:
        if self.n_bb1 < self.n_bb2:
            raise ValueError(
                f"{self.net_id}: BB1 连接数必须 >= BB2 "
                f"({self.bb1_sym}={self.n_bb1} < {self.bb2_sym}={self.n_bb2})"
            )
        if self.default_stacking not in self.stackings:
            raise ValueError(f"{self.net_id}: default_stacking 不在 stackings 内")


# ----------------------------------------------------------------------------
# 注册表本体：只有这里列出的组合可以被枚举
# ----------------------------------------------------------------------------
_NETS: Tuple[NetSpec, ...] = (
    # ---- T3 + L2 -> hcb（蜂窝，最经典的 TAPB-PDA / COF-1 体系） ----
    NetSpec("hcb_a", "HCB_A", "create_hcb_a_structure", "T3", "L2",
            note="3,2-c 蜂窝网；无 c_parameter_base，c 由 set_c_parameter 后处理"),
    # ---- T3 + T3 -> hcb（双三角节点自互补） ----
    NetSpec("hcb", "HCB", "create_hcb_structure", "T3", "T3",
            note="两个 C3 节点互为连接子"),
    # ---- S4 + L2 -> sql（正方，卟啉/酞菁类最常用） ----
    NetSpec("sql_a", "SQL_A", "create_sql_a_structure", "S4", "L2",
            note="4,2-c 正方网；c 参数由 c_parameter_base 控制"),
    # ---- S4 + L2 -> fxt（矩形，非对称 4-c 节点的正确落点） ----
    NetSpec("fxt_a", "FXT_A", "create_fxt_a_structure", "S4", "L2",
            note="4,2-c 矩形网；ETTA / PyTTA 这类非对称矩形节点的首选"),
    # ---- S4 + S4 -> sql ----
    NetSpec("sql", "SQL", "create_sql_structure", "S4", "S4",
            note="两个 4-c 节点"),
    # ---- S4 + S4 -> fxt（矩形 4-c + 矩形 4-c） ----
    NetSpec("fxt", "FXT", "create_fxt_structure", "S4", "S4",
            note="两个矩形 4-c 节点；pcb 内部按 R4 处理"),
    # ---- H6 + T3 -> kgd（3,6-c 双节点） ----
    NetSpec("kgd", "KGD", "create_kgd_structure", "H6", "T3",
            note="HPB 六苯醛类 H6 节点 + 三胺"),
    # ---- H6 + L2 -> hxl（6,2-c 六方） ----
    NetSpec("hxl_a", "HXL_A", "create_hxl_a_structure", "H6", "L2",
            note="6,2-c 六方；无 c_parameter_base"),
)

#: 默认关闭的 3D 拓扑（v2.0 约定：只留开关，不参与候选枚举）
_NETS_3D: Tuple[NetSpec, ...] = (
    NetSpec("bor", "BOR", "create_bor_structure", "D4", "T3",
            stackings=("1",), default_stacking="1", enabled=False,
            note="3D；D4 + T3"),
    NetSpec("dia", "DIA", "create_dia_structure", "D4", "D4",
            stackings=("1",), default_stacking="1", enabled=False,
            note="3D；需穿插度 1..5"),
    NetSpec("lon", "LON", "create_lon_structure", "D4", "D4",
            stackings=("1",), default_stacking="1", enabled=False,
            note="3D；需穿插度 1..5"),
)

#: 0.0.9.4 实测在上游无法成功建胞的 net（换任何基元都 BondLenghError）。
#: 不是"我们不用"，是"它不能用"；升级 pcb 后需重新验证再解封。
UPSTREAM_BLOCKED: FrozenSet[str] = frozenset()  # 0.0.9.4 复测：FXT_A 对 PORP/PTCA/TBPR/PHPR 可建胞

ALL_NETS: Tuple[NetSpec, ...] = _NETS + _NETS_3D
BY_ID: Dict[str, NetSpec] = {n.net_id: n for n in ALL_NETS}


def _pair_key(a: str, b: str) -> FrozenSet[str]:
    k = frozenset((a, b))
    if len(k) == 1:  # 同种对称性，frozenset 会塌成一个元素，这里补回来
        return frozenset((a, b, a))
    return k


_BY_PAIR: Dict[FrozenSet[str], List[NetSpec]] = {}
for _n in ALL_NETS:
    _BY_PAIR.setdefault(_pair_key(_n.bb1_sym, _n.bb2_sym), []).append(_n)


def resolve_nets(sym_a: str, sym_b: str, *, include_3d: bool = False,
                 include_disabled: bool = False) -> List[NetSpec]:
    """返回 (sym_a, sym_b) 组合下**真实可实现**的拓扑列表。

    返回空列表意味着「化学上可能，但 pyCOFBuilder 未实现」——调用方必须
    显式报错或跳过，不能猜测替代方案。
    """
    for s in (sym_a, sym_b):
        if s not in CONNECTIVITY:
            raise ValueError(f"未知基元对称性 {s!r}；合法值 {BB_SYMMETRIES}")
    cands = _BY_PAIR.get(_pair_key(sym_a, sym_b), [])
    out: List[NetSpec] = []
    for n in cands:
        if not include_3d and not n.is_2d:
            continue
        if not include_disabled and not n.enabled:
            continue
        if not include_disabled and n.net_id in UPSTREAM_BLOCKED:
            continue
        out.append(n)
    return out


def unsupported_reason(sym_a: str, sym_b: str) -> str:
    """给候选生成器用的可读诊断信息。"""
    nets = resolve_nets(sym_a, sym_b, include_3d=True, include_disabled=True)
    if nets:
        active = [n.net_id for n in nets if n.enabled and n.is_2d]
        if not active:
            return (f"({sym_a},{sym_b}) 仅有 3D/禁用拓扑 "
                    f"{[n.net_id for n in nets]}，v2.0 不枚举")
        return ""
    return (f"({sym_a},{sym_b}) 在 pyCOFBuilder 中无对应建网方法；"
            f"需自研 net 生成器后才能启用")


def canonical_bb_order(sym_a: str, conn_a: int, mw_a: float, smiles_a: str,
                       sym_b: str, conn_b: int, mw_b: float, smiles_b: str
                       ) -> Tuple[bool, Tuple]:
    """规范排序：连接数降序 -> 分子量降序 -> Canonical SMILES 升序。

    返回 (是否需要交换, 排序键)。交换标志用于保证 pcb 的 BB1 永远是
    高连接度节点（pcb 命名法硬性要求），同时让 A/B 顺序确定，
    从而彻底消除置换不变性导致的索引错位。
    """
    key_a = (-conn_a, -round(float(mw_a), 3), smiles_a or "")
    key_b = (-conn_b, -round(float(mw_b), 3), smiles_b or "")
    return (key_b < key_a), (key_a, key_b)


def stacking_candidates(net: NetSpec, *, include_optional: bool = False
                        ) -> Tuple[str, ...]:
    if include_optional:
        return tuple(net.stackings) + STACKING_2D_OPTIONAL
    return tuple(net.stackings)


def export_table() -> List[Dict[str, object]]:
    """供 run_w1_spike.py --list-topologies 打印 / 落盘 JSON 用。"""
    return [
        {
            "net_id": n.net_id,
            "net_token": n.net_token,
            "builder": n.builder,
            "pair": f"({n.bb1_sym},{n.bb2_sym})",
            "connectivity": f"({n.n_bb1},{n.n_bb2})",
            "stackings": list(n.stackings),
            "default": n.default_stacking,
            "enabled": n.enabled,
            "note": n.note,
        }
        for n in ALL_NETS
    ]


if __name__ == "__main__":  # 快速自检，不需要 pcb
    import json
    print(json.dumps(export_table(), ensure_ascii=False, indent=2))
    for a, b in [("T3", "L2"), ("S4", "L2"), ("H6", "T3"), ("T3", "S4"), ("D4", "L2")]:
        ids = [n.net_id for n in resolve_nets(a, b)]
        print(f"{a}+{b}: {ids or unsupported_reason(a, b)}")