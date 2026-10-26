"""Place two flat four-port monomers on a CIF's A/B sql nodes.

This is an initial heavy-atom *placement*, not a geometry optimisation.
It imports only the other files in mini_sql_lab and installed packages.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import spglib
from rdkit import Chem
from scipy.optimize import least_squares, linear_sum_assignment

from .groups import group_symbol, operations, read_template, subgroup_tree
from .input import Monomer, parse_pair, rigid_aromatic_core


class PlacementError(ValueError):
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


def _spin(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def _same_site(first: np.ndarray, second: np.ndarray, tolerance=1e-5) -> bool:
    difference = first - second
    return np.linalg.norm(difference - np.rint(difference)) < tolerance


def _site_rotations(template, group: int, site: str) -> tuple[np.ndarray, ...]:
    inverse = np.linalg.inv(template.cell)
    point = template.sites[site]
    if not all(_same_site(op.operate(point), point)
               for op in operations(group)):
        raise ValueError(f"group #{group} moves site {site} away from itself")
    return tuple(template.cell @ op.rotation_matrix @ inverse
                 for op in operations(group))


def _align(monomer: Monomer, rotations: tuple[np.ndarray, ...],
           preferred_degrees: float) -> tuple[np.ndarray, float]:
    """Search in-plane orientations, then average graph-equivalent images."""
    maps = np.array(monomer.automorphisms, dtype=int)
    if not len(maps):
        raise ValueError("monomer has no port-preserving graph automorphism")
    best = None
    for degree in range(0, 360, 2):
        rotated = monomer.positions @ np.block([
            [_spin(degree).T, np.zeros((2, 1))],
            [np.zeros((1, 2)), np.ones((1, 1))],
        ])
        worst = 0.0
        chosen = []
        for rotation in rotations:
            transformed = rotated @ rotation.T
            errors = np.mean(np.sum(
                (rotated[maps] - transformed[None, :, :]) ** 2, axis=2
            ), axis=1)
            index = int(np.argmin(errors))
            worst = max(worst, float(np.sqrt(errors[index])))
            chosen.append(maps[index])
        angular_distance = abs((degree - preferred_degrees + 180) % 360 - 180)
        score = worst + 0.003 * np.deg2rad(angular_distance)
        if best is None or score < best[0]:
            best = (score, worst, rotated, chosen)
    _, rms, rotated, chosen = best
    averaged = np.zeros_like(rotated)
    for rotation, mapping in zip(rotations, chosen):
        averaged[mapping] += rotated @ rotation.T
    averaged /= len(rotations)
    if any(np.max(np.linalg.norm(averaged[mapping] -
                                 averaged @ rotation.T, axis=1)) > 1e-4
           for rotation, mapping in zip(rotations, chosen)):
        raise ValueError("graph maps do not form a consistent site-symmetry action")
    return averaged, rms


def _port_order(monomer: Monomer, local: np.ndarray,
                directions: np.ndarray) -> tuple[int, ...]:
    points = local[list(monomer.ports), :2]
    unit_points = points / np.linalg.norm(points, axis=1)[:, None]
    unit_directions = directions[:, :2] / np.linalg.norm(
        directions[:, :2], axis=1)[:, None]
    rows, columns = linear_sum_assignment(-(unit_points @ unit_directions.T))
    result = [0] * 4
    for row, column in zip(rows, columns):
        result[int(column)] = int(row)
    return tuple(result)


def _four_cycles(shifts: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    cycles = []
    for first, fourth in combinations(range(4), 2):
        for second, third in product(range(4), repeat=2):
            translation = shifts[first] - shifts[second]
            if np.any(translation) and np.array_equal(
                translation, shifts[fourth] - shifts[third]
            ):
                cycles.append((first, second, third, fourth))
    return cycles


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.dot(first, second) / (
        np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _pore_error(first: Monomer, second: Monomer, first_local: np.ndarray,
                second_local: np.ndarray, first_order: tuple[int, ...],
                second_order: tuple[int, ...], shifts: list[np.ndarray]) -> float:
    first_points = first_local[list(first.ports), :2]
    second_points = second_local[list(second.ports), :2]
    errors = []
    for e0, e1, e2, e3 in _four_cycles(shifts):
        angles = (
            _angle(first_points[first_order[e0]], first_points[first_order[e3]]),
            _angle(second_points[second_order[e0]], second_points[second_order[e1]]),
            _angle(first_points[first_order[e1]], first_points[first_order[e2]]),
            _angle(second_points[second_order[e2]], second_points[second_order[e3]]),
        )
        errors.append(abs(sum(angles) - 360.0))
    if not errors:
        raise ValueError("template has no sql four-cycle")
    return max(errors)


def _internal_bond_change(monomer: Monomer, local: np.ndarray) -> float:
    """Catch conformer projections that collapse covalent bonds."""
    return max((abs(
        np.linalg.norm(local[second] - local[first])
        - np.linalg.norm(monomer.positions[second] - monomer.positions[first])
    ) for first, second in monomer.bonds), default=0.0)


def _fit_cell(template, group: int, first_local: np.ndarray,
              second_local: np.ndarray, first: Monomer, second: Monomer,
              first_order: tuple[int, ...], second_order: tuple[int, ...],
              directions: np.ndarray) -> tuple[np.ndarray, float]:
    square = any(abs(op.rotation_matrix[0, 1]) > 0.5 or
                 abs(op.rotation_matrix[1, 0]) > 0.5
                 for op in operations(group))
    start = np.diag(template.cell)[:2]
    first_points = first_local[list(first.ports), :2]
    second_points = second_local[list(second.ports), :2]

    def residual(values):
        a = values[0]
        b = a if square else values[1]
        result = []
        for index, direction in enumerate(directions):
            carbon = first_points[first_order[index]]
            nitrogen = second_points[second_order[index]]
            outward = carbon / np.linalg.norm(carbon)
            bond = direction[:2] * [a, b] + nitrogen - carbon
            result.extend(bond - 1.28 * outward)
        return result

    initial = [float(np.mean(start))] if square else list(start)
    fit = least_squares(residual, initial,
                        bounds=(np.array(initial) * 0.5,
                                np.array(initial) * 1.8),
                        max_nfev=300)
    cell = np.diag([float(fit.x[0]),
                    float(fit.x[0] if square else fit.x[1]),
                    float(template.cell[2, 2])])
    return cell, float(np.linalg.norm(fit.fun))


def _canonical_bond(first: int, second: int,
                    shift: np.ndarray) -> tuple[int, int, tuple[int, int, int]]:
    if first > second:
        first, second, shift = second, first, -shift
    return first, second, tuple(int(item) for item in shift)


def _full_structure(template, cell: np.ndarray, first: Monomer,
                    second: Monomer, first_local: np.ndarray,
                    second_local: np.ndarray, first_order: tuple[int, ...],
                    second_order: tuple[int, ...]):
    inverse = np.linalg.inv(cell)
    offset = len(first.elements)
    fractional = np.mod(np.vstack([
        first_local + cell @ template.sites["A"],
        second_local + cell @ template.sites["B"],
    ]) @ inverse.T, 1.0)
    elements = first.elements + second.elements
    roles = ("A",) * offset + ("B",) * len(second.elements)
    bonds = set()

    def add_bond(atom_a, atom_b, desired_delta):
        raw = fractional[atom_b] - fractional[atom_a]
        shift = np.rint(desired_delta - raw).astype(int)
        if np.linalg.norm(cell @ (raw + shift - desired_delta)) > 1e-5:
            raise ValueError("bond image cannot be matched to template")
        bonds.add(_canonical_bond(atom_a, atom_b, shift))

    for monomer, local, atom_offset in (
        (first, first_local, 0), (second, second_local, offset)
    ):
        for atom_a, atom_b in monomer.bonds:
            add_bond(atom_a + atom_offset, atom_b + atom_offset,
                     inverse @ (local[atom_b] - local[atom_a]))
    imine_distances = []
    for index, (_, _, shift) in enumerate(template.edges):
        carbon = first.ports[first_order[index]]
        nitrogen_local = second.ports[second_order[index]]
        nitrogen = offset + nitrogen_local
        desired = (inverse @ (second_local[nitrogen_local] - first_local[carbon])
                   + template.sites["B"] + np.array(shift) - template.sites["A"])
        add_bond(carbon, nitrogen, desired)
        imine_distances.append(float(np.linalg.norm(cell @ desired)))
    return fractional, elements, roles, bonds, imine_distances


def _symmetry_maps(group: int, fractional: np.ndarray,
                   elements: tuple[str, ...], roles: tuple[str, ...],
                   bonds: set[tuple]) -> list[list[int]]:
    """Verify each fixed group operation maps atoms AND periodic bonds."""
    all_maps = []
    for op in operations(group):
        mapping, lattice_shifts = [], []
        for index, position in enumerate(fractional):
            transformed = op.operate(position)
            choices = [
                other for other, target in enumerate(fractional)
                if elements[other] == elements[index]
                and roles[other] == roles[index]
                and _same_site(transformed, target, 1e-4)
            ]
            if len(choices) != 1:
                raise ValueError("group operation does not uniquely map all atoms")
            mapping.append(choices[0])
            lattice_shifts.append(np.rint(
                transformed - fractional[choices[0]]
            ).astype(int))
        if len(set(mapping)) != len(mapping):
            raise ValueError("group operation merges distinct atoms")
        transformed_bonds = {
            _canonical_bond(
                mapping[atom_a], mapping[atom_b],
                lattice_shifts[atom_b]
                + np.rint(op.rotation_matrix).astype(int) @ np.array(shift)
                - lattice_shifts[atom_a],
            )
            for atom_a, atom_b, shift in bonds
        }
        if transformed_bonds != bonds:
            raise ValueError("group operation changes the periodic chemical graph")
        all_maps.append(mapping)
    return all_maps


def _write_cif(path: Path, group: int, cell: np.ndarray,
               fractional: np.ndarray, elements: tuple[str, ...],
               bonds: set[tuple], maps: list[list[int]]) -> None:
    ops = operations(group)
    source_of = [None] * len(elements)
    representatives = []
    for index in range(len(elements)):
        if source_of[index] is not None:
            continue
        source = len(representatives)
        representatives.append(index)
        for op_index, mapping in enumerate(maps):
            image = mapping[index]
            if source_of[image] is None:
                source_of[image] = (source, op_index)
    if any(item is None for item in source_of):
        raise ValueError("asymmetric-unit expansion is incomplete")
    labels = [f"{elements[index]}{source}" for source, index
              in enumerate(representatives)]
    lines = [
        "# Initial heavy-atom sql placement; H omitted; NOT optimised",
        f"# declared group #{group} {group_symbol(group)}",
        f"data_mini_sql_{group}",
        f"_space_group_IT_number {group}",
        f"_space_group_name_H-M_alt '{group_symbol(group)}'",
        f"_cell_length_a {cell[0, 0]:.8f}",
        f"_cell_length_b {cell[1, 1]:.8f}",
        f"_cell_length_c {cell[2, 2]:.8f}",
        "_cell_angle_alpha 90",
        "_cell_angle_beta 90",
        "_cell_angle_gamma 90",
        "loop_",
        " _space_group_symop_id",
        " _space_group_symop_operation_xyz",
    ]
    for index, op in enumerate(ops, 1):
        lines.append(f" {index} '{op.as_xyz_str()}'")
    lines += [
        "loop_",
        " _atom_site_type_symbol",
        " _atom_site_label",
        " _atom_site_symmetry_multiplicity",
        " _atom_site_fract_x",
        " _atom_site_fract_y",
        " _atom_site_fract_z",
        " _atom_site_occupancy",
    ]
    for source, index in enumerate(representatives):
        orbit = {mapping[index] for mapping in maps}
        x, y, z = fractional[index]
        lines.append(f" {elements[index]} {labels[source]} {len(orbit)} "
                     f"{x:.8f} {y:.8f} {z:.8f} 1")
    lines += [
        "loop_",
        " _geom_bond_atom_site_label_1",
        " _geom_bond_atom_site_label_2",
        " _geom_bond_distance",
        " _geom_bond_site_symmetry_1",
        " _geom_bond_site_symmetry_2",
    ]
    for atom_a, atom_b, shift in sorted(bonds):
        source_a, op_a = source_of[atom_a]
        source_b, op_b = source_of[atom_b]
        generated_a = ops[op_a].operate(fractional[representatives[source_a]])
        generated_b = ops[op_b].operate(fractional[representatives[source_b]])
        move_a = np.rint(fractional[atom_a] - generated_a).astype(int)
        move_b = np.rint(fractional[atom_b] - generated_b).astype(int)
        move_b += np.array(shift)
        if any(abs(int(item)) > 4 for item in (*move_a, *move_b)):
            raise ValueError("CIF bond image shift exceeds one-digit encoding")
        code_a = "".join(str(5 + int(item)) for item in move_a)
        code_b = "".join(str(5 + int(item)) for item in move_b)
        distance = np.linalg.norm(
            cell @ (fractional[atom_b] + np.array(shift) - fractional[atom_a]))
        lines.append(f" {labels[source_a]} {labels[source_b]} {distance:.5f} "
                     f"{op_a + 1}_{code_a} {op_b + 1}_{code_b}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(config: dict, group: int, output: Path,
          max_symmetry_rms: float = 1.2,
          max_bond_change: float = 0.25) -> dict:
    template = read_template(config["template_cif"])
    tree = subgroup_tree(template.root_group)
    if group not in {item["number"] for item in tree["groups"]}:
        raise ValueError(f"#{group} is not in this template's same-cell subgroup tree")
    specs = config["monomers"]
    first, second = parse_pair(
        specs["A"]["smiles"], specs["B"]["smiles"],
        tuple(specs["A"]["original_reactive_atom_indices"]),
        tuple(specs["B"]["original_reactive_atom_indices"]),
    )
    if first.smiles != specs["A"]["smiles"]:
        raise ValueError("input report has swapped/invalid monomer roles")
    first_rotations = _site_rotations(template, group, "A")
    second_rotations = _site_rotations(template, group, "B")
    shifts = [np.array(edge[2], dtype=int) for edge in template.edges]
    directions = np.array([
        template.sites["B"] + shift - template.sites["A"] for shift in shifts
    ])
    candidates = []
    rejected = []
    for first_turn, second_turn in product((0, 90, 180, 270), repeat=2):
        try:
            first_local, first_rms = _align(first, first_rotations, first_turn)
            second_local, second_rms = _align(second, second_rotations, second_turn)
            if max(first_rms, second_rms) > max_symmetry_rms:
                raise ValueError(f"initial symmetry mismatch "
                                 f"{max(first_rms, second_rms):.3f} Å "
                                 f"> {max_symmetry_rms:.3f} Å")
            bond_change = max(_internal_bond_change(first, first_local),
                              _internal_bond_change(second, second_local))
            first_order = _port_order(first, first_local, directions)
            second_order = _port_order(second, second_local, -directions)
            pore_error = _pore_error(
                first, second, first_local, second_local,
                first_order, second_order, shifts,
            )
            if (rigid_aromatic_core(first) and rigid_aromatic_core(second)
                    and pore_error > 20.0):
                raise ValueError(f"rigid planar four-cycle error {pore_error:.1f}°")
            if bond_change > max_bond_change:
                raise ValueError(f"site-symmetry projection changes a monomer "
                                 f"bond by {bond_change:.3f} Å > "
                                 f"{max_bond_change:.3f} Å; a constrained "
                                 "optimizer or other conformer is needed")
            cell, fit_error = _fit_cell(
                template, group, first_local, second_local, first, second,
                first_order, second_order, directions,
            )
            fractional, elements, roles, bonds, imine = _full_structure(
                template, cell, first, second, first_local, second_local,
                first_order, second_order,
            )
            if max(abs(value - 1.28) for value in imine) > 0.35:
                raise ValueError("initial imine C=N distance differs by >0.35 Å")
            maps = _symmetry_maps(group, fractional, elements, roles, bonds)
            candidates.append((
                (round(fit_error, 6), round(pore_error, 6),
                 first_turn, second_turn),
                cell, fractional, elements, bonds, maps,
                first_order, second_order, first_rms, second_rms,
                pore_error, imine, bond_change,
            ))
        except ValueError as error:
            rejected.append({"orientation": [first_turn, second_turn],
                             "reason": str(error)})
    if not candidates:
        raise PlacementError(
            "no compatible initial placement; reasons: "
            + "; ".join(dict.fromkeys(item["reason"] for item in rejected)),
            rejected,
        )
    (score, cell, fractional, elements, bonds, maps, first_order, second_order,
     first_rms, second_rms, pore_error, imine, bond_change) = min(
        candidates, key=lambda item: item[0])
    _write_cif(output, group, cell, fractional, elements, bonds, maps)
    numbers = [Chem.GetPeriodicTable().GetAtomicNumber(element)
               for element in elements]
    detected = spglib.get_symmetry_dataset(
        (cell.T, fractional, numbers), symprec=1e-3,
    )
    return {
        "schema": "mini_sql_placement_v1",
        "cif": str(output),
        "declared_group": group,
        "detected_group_for_initial_heavy_atoms": (
            int(detected.number) if detected is not None else None
        ),
        "orientation_degrees": list(score[2:]),
        "port_mapping_A": list(first_order),
        "port_mapping_B": list(second_order),
        "symmetry_mapping_rms_angstrom": [first_rms, second_rms],
        "maximum_internal_bond_change_angstrom": bond_change,
        "maximum_four_cycle_angle_error_degrees": pore_error,
        "imine_bond_lengths_angstrom": imine,
        "cell_lengths_angstrom": list(np.diag(cell)),
        "heavy_atoms_in_cell": len(elements),
        "bonds_in_cell": len(bonds),
        "valid_placements_found": len(candidates),
        "rejected_placements": rejected,
        "warning": "Heavy-atom initial placement only. No H, force field, "
                   "cell/atomic relaxation, or experimental validation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON written by python -m mini_sql_lab.input")
    parser.add_argument("--group", type=int, required=True,
                        help="one group number in the reported same-cell tree")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("placed.cif"))
    parser.add_argument("--max-symmetry-rms", type=float, default=1.2,
                        help="heuristic initial conformer precheck, Å")
    parser.add_argument("--max-bond-change", type=float, default=0.25,
                        help="maximum internal bond change on symmetry projection, Å")
    args = parser.parse_args()
    config = json.loads(args.input.read_text(encoding="utf-8"))
    template_path = Path(config["template_cif"])
    if not template_path.is_absolute():
        config["template_cif"] = str(
            (args.input.resolve().parent / template_path).resolve()
        )
    report_path = args.out.with_name(args.out.stem + "_report.json")
    try:
        report = build(config, args.group, args.out, args.max_symmetry_rms,
                       args.max_bond_change)
    except ValueError as error:
        report = {
            "schema": "mini_sql_placement_v1",
            "cif": None,
            "declared_group": args.group,
            "reason": str(error),
            "rejected_placements": (
                error.attempts if isinstance(error, PlacementError) else []
            ),
            "preexisting_file_at_requested_cif_path": args.out.exists(),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"no CIF generated: {error}")
        print(f"failure report: {report_path}")
        return 2
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"initial CIF: {args.out}")
    print(f"report: {report_path}")
    print(f"group #{args.group}; four-cycle error "
          f"{report['maximum_four_cycle_angle_error_degrees']:.3f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
