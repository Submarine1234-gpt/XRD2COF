"""SMILES -> four reaction ports, graph-port symmetry and CIF group tree."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .groups import read_template, subgroup_tree


PATTERNS = {
    "aldehyde": Chem.MolFromSmarts("[CX3H1](=O)[#6]"),
    "amine": Chem.MolFromSmarts("[NX3;H2][#6]"),
}


@dataclass(frozen=True)
class Monomer:
    smiles: str
    kind: str
    elements: tuple[str, ...]
    positions: np.ndarray  # 2D heavy-atom drawing in Å, centred at four ports
    bonds: tuple[tuple[int, int], ...]
    ports: tuple[int, ...]  # atom indices after reactive carbonyl O removal
    original_ports: tuple[int, ...]  # indices in input SMILES
    automorphisms: tuple[tuple[int, ...], ...]
    aromatic_bonds: frozenset[tuple[int, int]]


def _sites(molecule: Chem.Mol, kind: str) -> tuple[int, ...]:
    return tuple(sorted(match[0] for match in
                        molecule.GetSubstructMatches(PATTERNS[kind])))


def parse_monomer(smiles: str, kind: str,
                  selected: tuple[int, ...] | None = None) -> Monomer:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    if any(atom.GetHybridization() == Chem.HybridizationType.SP3
           and atom.GetDegree() >= 3 for atom in molecule.GetAtoms()):
        raise ValueError("this planar demo does not place a 3D sp3 core")
    if kind not in PATTERNS:
        raise ValueError("only aldehyde + amine imine chemistry is supported")
    found = _sites(molecule, kind)
    selected = found if selected is None else selected
    if len(selected) != 4 or len(set(selected)) != 4 or not set(selected) <= set(found):
        raise ValueError(f"{kind} needs exactly four distinct reactive ports; "
                         f"found {list(found)}, selected {list(selected)}")
    AllChem.Compute2DCoords(molecule)
    drawing = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    leaving = set()
    if kind == "aldehyde":
        for port in selected:
            oxygen = [neighbor.GetIdx() for neighbor in
                      molecule.GetAtomWithIdx(port).GetNeighbors()
                      if neighbor.GetSymbol() == "O" and
                      molecule.GetBondBetweenAtoms(port, neighbor.GetIdx())
                      .GetBondType() == Chem.BondType.DOUBLE]
            if len(oxygen) != 1:
                raise ValueError(f"aldehyde carbon {port} has no unique oxygen")
            leaving.add(oxygen[0])
    kept = [index for index in range(molecule.GetNumAtoms()) if index not in leaving]
    old_to_new = {old: new for new, old in enumerate(kept)}
    editor = Chem.RWMol(molecule)
    for index in sorted(leaving, reverse=True):
        editor.RemoveAtom(index)
    product = editor.GetMol()
    Chem.SanitizeMol(product)
    positions = drawing[kept].copy()
    positions[:, 2] = 0.0
    ports = tuple(old_to_new[index] for index in selected)
    positions -= positions[list(ports)].mean(axis=0)
    bonds = tuple(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                  for bond in product.GetBonds())
    aromatic = frozenset(
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in product.GetBonds() if bond.GetIsAromatic()
    )
    mappings = product.GetSubstructMatches(product, uniquify=False,
                                           maxMatches=4097)
    if len(mappings) > 4096:
        raise ValueError("more than 4096 graph automorphisms; search truncated")
    automorphisms = tuple(tuple(mapping) for mapping in mappings
                          if {mapping[index] for index in ports} == set(ports))
    return Monomer(smiles, kind,
                   tuple(atom.GetSymbol() for atom in product.GetAtoms()),
                   positions, bonds, ports, selected, automorphisms, aromatic)


def parse_pair(first_smiles: str, second_smiles: str,
               first_ports: tuple[int, ...] | None = None,
               second_ports: tuple[int, ...] | None = None) -> tuple[Monomer, Monomer]:
    errors = []
    for aldehyde_smiles, amine_smiles, aldehyde_ports, amine_ports in (
        (first_smiles, second_smiles, first_ports, second_ports),
        (second_smiles, first_smiles, second_ports, first_ports),
    ):
        try:
            return (parse_monomer(aldehyde_smiles, "aldehyde", aldehyde_ports),
                    parse_monomer(amine_smiles, "amine", amine_ports))
        except ValueError as error:
            errors.append(str(error))
    raise ValueError("inputs are not a four-aldehyde/four-amine pair: "
                     + " | ".join(errors))


def rigid_aromatic_core(monomer: Monomer) -> bool:
    """All four ports directly attach to one aromatic-bond component."""
    adjacent = {index: set() for index in range(len(monomer.elements))}
    aromatic = {index: set() for index in range(len(monomer.elements))}
    for first, second in monomer.bonds:
        adjacent[first].add(second)
        adjacent[second].add(first)
        if (first, second) in monomer.aromatic_bonds:
            aromatic[first].add(second)
            aromatic[second].add(first)
    anchors = []
    for port in monomer.ports:
        nearby = [index for index in adjacent[port] if aromatic[index]]
        if len(nearby) != 1:
            return False
        anchors.append(nearby[0])
    visited, pending = {anchors[0]}, [anchors[0]]
    while pending:
        for neighbor in aromatic[pending.pop()]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) >= 6 and set(anchors) <= visited


def monomer_report(monomer: Monomer) -> dict:
    """Graph symmetry of selected ports; not a 3D molecular point group."""
    points = monomer.positions[list(monomer.ports), :2]
    angles = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * np.pi)
    order = tuple(int(item) for item in np.argsort(angles))
    sorted_angles = np.sort(angles)
    gaps = np.degrees(np.diff(np.r_[sorted_angles,
                                    sorted_angles[0] + 2 * np.pi]))
    positions = {atom: index for index, atom in enumerate(monomer.ports)}
    permutations = {
        tuple(positions[mapping[atom]] for atom in monomer.ports)
        for mapping in monomer.automorphisms
    }

    def cyclic_permutation(shift: int, reflect: bool) -> tuple[int, ...]:
        result = [0] * 4
        for index, port_index in enumerate(order):
            result[port_index] = order[(shift - index if reflect
                                        else shift + index) % 4]
        return tuple(result)

    rotations = [90 * shift for shift in range(4)
                 if cyclic_permutation(shift, False) in permutations]
    reflections = sum(cyclic_permutation(shift, True) in permutations
                      for shift in range(4))
    return {
        "smiles": monomer.smiles,
        "role": monomer.kind,
        "original_reactive_atom_indices": list(monomer.original_ports),
        "heavy_atom_count_after_reaction": len(monomer.elements),
        "port_cyclic_gaps_degrees_from_2d_drawing":
            [round(float(value), 3) for value in gaps],
        "port_graph_rotations_degrees": rotations,
        "port_graph_reflection_count": reflections,
        "port_graph_permutations": sorted([list(item) for item in permutations]),
        "all_ports_on_one_rigid_aromatic_core": rigid_aromatic_core(monomer),
        "symmetry_meaning": "heavy-atom graph automorphisms after removing "
                            "reactive carbonyl O, preserving selected ports; "
                            "not a measured 3D point group",
    }


def _port_list(raw: str | None) -> tuple[int, ...] | None:
    return None if raw is None else tuple(int(item.strip()) for item in raw.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-smiles", required=True)
    parser.add_argument("--second-smiles", required=True)
    parser.add_argument("--first-ports", help="optional original atom indices, comma-separated")
    parser.add_argument("--second-ports", help="optional original atom indices, comma-separated")
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).with_name("sql_nodes.cif"))
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("example_input.json"))
    args = parser.parse_args()
    try:
        first, second = parse_pair(args.first_smiles, args.second_smiles,
                                   _port_list(args.first_ports),
                                   _port_list(args.second_ports))
        template = read_template(args.template)
    except ValueError as error:
        print(f"input rejected: {error}")
        return 2
    report = {
        "schema": "mini_sql_input_v1",
        "template_cif": os.path.relpath(args.template.resolve(),
                                        args.out.parent.resolve()),
        "monomers": {"A": monomer_report(first), "B": monomer_report(second)},
        "space_group_tree": subgroup_tree(template.root_group),
        "limitations": "2D four-port sql; graph-port symmetry only; no conformer "
                       "ensemble or geometry optimisation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"input report: {args.out}")
    for role in ("A", "B"):
        item = report["monomers"][role]
        print(f"{role} {item['role']}: ports {item['original_reactive_atom_indices']}, "
              f"graph rotations {item['port_graph_rotations_degrees']}, "
              f"2D gaps {item['port_cyclic_gaps_degrees_from_2d_drawing']}")
    print(f"CIF root #{template.root_group}; "
          f"{len(report['space_group_tree']['groups'])} same-cell groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
