"""Read a tiny node CIF and derive its same-cell subgroup tree.

This file imports no code from the main COF project. Group operations come
from the installed pymatgen database, in its first/standard setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import gemmi
import numpy as np
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.groups import SpaceGroup


@dataclass(frozen=True)
class Template:
    path: Path
    root_group: int
    cell: np.ndarray  # columns are lattice vectors, Å
    sites: dict[str, np.ndarray]
    edges: tuple[tuple[str, str, tuple[int, int, int]], ...]


def _number(value: str) -> float:
    return float(str(value).strip("'\"").split("(")[0])


@lru_cache(maxsize=230)
def group_symbol(number: int) -> str:
    if not 1 <= number <= 230:
        raise ValueError("space-group number must be 1..230")
    return next(
        entry["hermann_mauguin"] for entry in SpaceGroup.SYMM_OPS
        if entry["number"] == number
    )


@lru_cache(maxsize=230)
def operations(number: int) -> tuple[SymmOp, ...]:
    return tuple(sorted(SpaceGroup(group_symbol(number)).symmetry_ops,
                        key=lambda item: item.as_xyz_str()))


def operation_key(operation: SymmOp) -> tuple:
    rotation = tuple(np.rint(operation.rotation_matrix).astype(int).ravel())
    translation = np.mod(operation.translation_vector, 1.0)
    translation = tuple(round(float(item) % 1.0, 6) % 1.0
                        for item in translation)
    return rotation, translation


@lru_cache(maxsize=230)
def operation_set(number: int) -> frozenset[tuple]:
    return frozenset(operation_key(item) for item in operations(number))


def read_template(path: str | Path) -> Template:
    path = Path(path)
    block = gemmi.cif.read_file(str(path)).sole_block()
    raw_group = (block.find_value("_space_group_IT_number")
                 or block.find_value("_symmetry_Int_Tables_number"))
    if not raw_group:
        raise ValueError("template CIF has no space-group number")
    group = int(_number(raw_group))
    lengths = [_number(block.find_value(f"_cell_length_{axis}"))
               for axis in "abc"]
    angles = [_number(block.find_value(f"_cell_angle_{axis}"))
              for axis in ("alpha", "beta", "gamma")]
    if any(abs(angle - 90.0) > 1e-5 for angle in angles):
        raise ValueError("this minimal builder accepts orthogonal cells only")
    if any(length <= 0 for length in lengths):
        raise ValueError("cell lengths must be positive")
    labels = list(block.find_values("_atom_site_label"))
    coordinates = [list(block.find_values(f"_atom_site_fract_{axis}"))
                   for axis in "xyz"]
    sites = {
        str(label): np.array([_number(column[index]) for column in coordinates])
        for index, label in enumerate(labels)
    }
    if set(sites) != {"A", "B"}:
        raise ValueError("template needs exactly two abstract sites labelled A and B")
    columns = [list(block.find_values(f"_mini_edge_{name}")) for name in
               ("from", "to", "shift_x", "shift_y", "shift_z")]
    if not columns[0] or any(len(column) != len(columns[0]) for column in columns):
        raise ValueError("template needs a complete _mini_edge_* loop")
    edges = tuple(
        (str(columns[0][index]), str(columns[1][index]),
         tuple(int(_number(columns[axis][index])) for axis in (2, 3, 4)))
        for index in range(len(columns[0]))
    )
    if len(edges) != 4 or any(start != "A" or end != "B"
                              for start, end, _ in edges):
        raise ValueError("this minimal sql builder needs four A-to-B edges")
    declared_ops = (list(block.find_values("_space_group_symop_operation_xyz"))
                    or list(block.find_values("_symmetry_equiv_pos_as_xyz")))
    if declared_ops:
        from_cif = {operation_key(SymmOp.from_xyz_str(str(item).strip("'\"")))
                    for item in declared_ops}
        if from_cif != operation_set(group):
            raise ValueError("CIF symmetry operations differ from pymatgen's "
                             "standard setting for this group")
    return Template(path, group, np.diag(lengths), sites, edges)


def subgroup_tree(root: int) -> dict:
    """Operation inclusion only: same basis, origin and standard setting."""
    root_ops = operation_set(root)
    candidates = {
        number: group_ops for number in range(1, 231)
        if (group_ops := operation_set(number)) <= root_ops
        and (number == root or group_ops != root_ops)
    }
    edges = [
        [parent, child]
        for parent, parent_ops in candidates.items()
        for child, child_ops in candidates.items()
        if child_ops < parent_ops
        and not any(child_ops < middle_ops < parent_ops
                    for middle_ops in candidates.values())
    ]
    return {
        "root": root,
        "source": f"pymatgen {version('pymatgen')} SpaceGroup.SYMM_OPS",
        "scope": "same cell basis, same origin, pymatgen standard settings",
        "groups": [
            {"number": number, "symbol": group_symbol(number),
             "operation_count": len(group_ops)}
            for number, group_ops in sorted(candidates.items(),
                key=lambda item: (-len(item[1]), item[0]))
        ],
        "edges": sorted(edges),
    }
