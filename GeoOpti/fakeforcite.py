"""Symmetry-preserving geometry optimization with GULP UFF and ASE BFGS."""

import argparse
from collections import Counter
from pathlib import Path
import tempfile

import numpy as np
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import check_symmetry
from ase.units import GPa
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from GeoOpti.gulp_uff import GulpUFF, load_uff_model


class StressBFGS(BFGS):
    """Require both the force criterion and an optional stress limit in GPa."""

    def __init__(self, *args, stress_atoms=None, smax=0.05, **kwargs):
        self.stress_atoms = stress_atoms
        self.smax = smax
        super().__init__(*args, **kwargs)

    def gradient_converged(self, gradient):
        if not super().gradient_converged(gradient):
            return False
        return (self.stress_atoms is None or self.smax is None
                or np.max(np.abs(self.stress_atoms.get_stress())) <= self.smax * GPa)


def checked_spacegroup(atoms, symprec, expected=None):
    dataset = check_symmetry(atoms, symprec=symprec)
    if dataset is None:
        raise RuntimeError("Could not identify the structure's space group.")
    declared = atoms.info.get("spacegroup")
    if declared is not None and declared.no != dataset.number:
        raise RuntimeError(f"CIF declares group {declared.no}, but geometry has group {dataset.number}.")
    if expected is not None and dataset.number != expected:
        raise RuntimeError(f"Space group changed from {expected} to {dataset.number}.")
    return dataset


def write_checked_cif(atoms, output, symprec, expected):
    """Check the serialized structure before replacing the final output."""
    structure = Structure(atoms.cell.array, atoms.get_chemical_symbols(),
                          atoms.get_scaled_positions())
    with tempfile.NamedTemporaryFile(suffix=".cif", dir=output.parent, delete=False) as file:
        candidate = Path(file.name)
    try:
        CifWriter(structure, symprec=symprec, angle_tolerance=-1,
                  refine_struct=False, significant_figures=12).write_file(candidate)
        restored = read(candidate)
        checked_spacegroup(restored, symprec, expected)
        if Counter(restored.get_chemical_symbols()) != Counter(atoms.get_chemical_symbols()):
            raise RuntimeError("Exported CIF changed the composition.")
        candidate.replace(output)
    finally:
        candidate.unlink(missing_ok=True)


class GeometryOptimizer:
    def __init__(self, structure_path, symprec=0.01, *, uff_model=None,
                 gulp_executable="gulp", uff_library=None, calculator=None, workdir=None):
        self.structure_path = Path(structure_path)
        self.symprec = symprec
        self.atoms = read(self.structure_path)
        for site in self.atoms.info.get("occupancy", {}).values():
            if len(site) != 1 or not np.isclose(next(iter(site.values())), 1.0):
                raise ValueError("Input has partial/mixed occupancy; prepare a fully occupied model.")
        if calculator is None:
            model_path = uff_model or self.structure_path.with_suffix(".uff.json")
            model = load_uff_model(model_path, self.structure_path, self.atoms)
            calculator = GulpUFF(
                model["symbols"], model["types"], model["bonds"], model.get("charges"),
                executable=gulp_executable, library=uff_library, directory=workdir,
            )
        self.atoms.calc = calculator

    def optimize(self, fmax=0.002, steps=1000, output_path=None, *,
                 optimize_cell=True, logfile=None, smax=0.05, maxstep=0.1):
        if fmax <= 0 or steps <= 0 or maxstep <= 0 or (smax is not None and smax <= 0):
            raise ValueError("Force/stress limits, steps and maxstep must be positive.")
        if output_path is None:
            stem = self.structure_path.stem.removesuffix("_prepared")
            output_path = self.structure_path.with_name(f"{stem}_optimized.cif")
        output = Path(output_path)
        initial = checked_spacegroup(self.atoms, self.symprec)

        calculator = self.atoms.calc
        if isinstance(calculator, GulpUFF):
            for group in set(initial.equivalent_atoms):
                indices = np.flatnonzero(initial.equivalent_atoms == group)
                if len({calculator.uff_types[i] for i in indices}) != 1:
                    raise ValueError("Symmetry-equivalent atoms have different UFF types.")

        self.atoms.set_constraint(FixSymmetry(self.atoms, symprec=self.symprec))
        target = FrechetCellFilter(self.atoms) if optimize_cell else self.atoms
        optimizer = StressBFGS(
            target, logfile=logfile or output.with_suffix(".log"), maxstep=maxstep,
            stress_atoms=self.atoms if optimize_cell else None, smax=smax,
        )
        if not optimizer.run(fmax=fmax, steps=steps):
            raise RuntimeError("Optimization did not converge; no output CIF was written.")
        final = checked_spacegroup(self.atoms, self.symprec, initial.number)

        if isinstance(calculator, GulpUFF):
            for i, j, order, image in calculator.bonds:
                if order == "resonant":
                    vector = (self.atoms.positions[j] + np.array(image) @ self.atoms.cell.array
                              - self.atoms.positions[i])
                    if not 1.2 <= np.linalg.norm(vector) <= 1.65:
                        raise RuntimeError(f"Unreasonable resonant bond {i}-{j}; CIF not written.")

        write_checked_cif(self.atoms, output, self.symprec, initial.number)
        force = np.linalg.norm(self.atoms.get_forces(), axis=1).max()
        print(f"Space group {final.number} ({final.international}); max force {force:.6f} eV/Angstrom")
        if optimize_cell:
            stress = np.max(np.abs(self.atoms.get_stress())) / GPa
            print(f"Max stress {stress:.6f} GPa")
        print(f"Saved {output}")
        return self.atoms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", nargs="?", default="Data/example_10_prepared.cif")
    parser.add_argument("--output")
    parser.add_argument("--uff-model", help="Defaults to the input's .uff.json sidecar")
    parser.add_argument("--gulp", default="gulp")
    parser.add_argument("--uff-library", help="Path to uff.lib")
    parser.add_argument("--workdir", help="Keep GULP intermediate files here for debugging")
    parser.add_argument("--fixed-cell", action="store_true")
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--fmax", type=float, default=0.002, help="eV/Angstrom")
    parser.add_argument("--smax", type=float, default=0.05, help="GPa")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--logfile", help="Defaults to the output's .log file")
    args = parser.parse_args()
    optimizer = GeometryOptimizer(
        args.structure, symprec=args.symprec, uff_model=args.uff_model,
        gulp_executable=args.gulp, uff_library=args.uff_library, workdir=args.workdir,
    )
    try:
        optimizer.optimize(fmax=args.fmax, steps=args.steps, output_path=args.output,
                           optimize_cell=not args.fixed_cell, logfile=args.logfile, smax=args.smax)
    finally:
        optimizer.atoms.calc.close()


if __name__ == "__main__":
    main()
