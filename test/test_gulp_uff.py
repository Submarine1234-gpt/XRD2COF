"""Focused checks against a real GULP installation (stdlib unittest runner)."""

import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import CalculationFailed
from ase.calculators.emt import EMT
from ase.cell import Cell
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import check_symmetry
from ase.units import GPa
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from GeoOpti.fakeforcite import GeometryOptimizer, StressBFGS
from GeoOpti.gulp_uff import GulpUFF


def benzene(radius=1.4):
    angles = np.arange(6) * np.pi / 3
    positions = [[r * np.cos(a) + 10, r * np.sin(a) + 10, 10]
                 for r in (radius, radius + 1.08) for a in angles]
    return Atoms("C6H6", positions=positions, cell=[20, 20, 20], pbc=True)


def benzene_model(atoms):
    """Explicit known benzene chemistry; find only its index/image mapping."""
    symbols = atoms.get_chemical_symbols()
    i_list, j_list, shifts = neighbor_list("ijS", atoms, 1.7)
    bonds = [dict(i=int(i), j=int(j), image=shift.tolist(),
                  order="resonant" if symbols[i] == symbols[j] == "C" else "single")
             for i, j, shift in zip(i_list, j_list, shifts) if i < j]
    return symbols, ["C_R" if s == "C" else "H_" for s in symbols], bonds


class InputValidationTests(unittest.TestCase):
    def test_partial_occupancy_needs_an_explicit_model_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.cif"
            structure = Structure(np.eye(3) * 10, [{"C": 0.7}], [[0, 0, 0]])
            CifWriter(structure).write_file(path)
            with self.assertRaisesRegex(ValueError, "partial/mixed occupancy"):
                GeometryOptimizer(path)


class ConvergenceTests(unittest.TestCase):
    def test_small_forces_do_not_hide_unrelaxed_cell_stress(self):
        atoms = bulk("Al", "fcc", a=4.06, cubic=True)
        atoms.calc = EMT()
        cell_filter = FrechetCellFilter(atoms)
        # A loose generalized-force criterion alone accepts this strained cell.
        loose = BFGS(cell_filter, logfile=None)
        self.assertTrue(loose.run(fmax=0.5, steps=0))
        self.assertGreater(np.max(np.abs(atoms.get_stress())) / GPa, 0.01)
        strict = StressBFGS(cell_filter, stress_atoms=atoms, smax=0.01,
                            logfile=None, maxstep=0.1)
        self.assertTrue(strict.run(fmax=0.5, steps=100))
        self.assertGreater(strict.nsteps, 0)
        self.assertLessEqual(np.max(np.abs(atoms.get_stress())) / GPa, 0.01)


@unittest.skipUnless(shutil.which(os.environ.get("GULP_EXECUTABLE", "gulp")),
                     "Set GULP_EXECUTABLE to the GULP binary")
class GulpIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def calculator(self, atoms, model=None):
        symbols, types, bonds = benzene_model(atoms) if model is None else model
        return GulpUFF(symbols, types, bonds,
                       executable=os.environ.get("GULP_EXECUTABLE", "gulp"),
                       directory=self.directory / "singlepoint")

    def test_forces_and_all_stress_components_match_energy_differences(self):
        atoms = benzene()
        atoms.positions[0] += [0.02, -0.01, 0.03]
        atoms.rotate(17, "z", rotate_cell=True)
        atoms.calc = self.calculator(atoms)
        reference_positions = atoms.positions.copy()
        reference_cell = atoms.cell.array.copy()
        analytic_force = atoms.get_forces()[0, 0]
        analytic_stress = atoms.get_stress()
        step = 1e-4
        atoms.positions[0, 0] += step
        plus = atoms.get_potential_energy()
        atoms.positions[0, 0] -= 2 * step
        minus = atoms.get_potential_energy()
        atoms.positions[:] = reference_positions
        self.assertAlmostEqual(analytic_force, -(plus - minus) / (2 * step), delta=5e-5)
        for voigt, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
            energies = []
            for sign in (1, -1):
                deformation = np.eye(3)
                deformation[i, j] += sign * step if i == j else sign * step / 2
                if i != j:
                    deformation[j, i] += sign * step / 2
                atoms.set_cell(reference_cell @ deformation)
                atoms.positions[:] = reference_positions @ deformation
                energies.append(atoms.get_potential_energy())
            volume = abs(np.linalg.det(reference_cell))
            numerical = (energies[0] - energies[1]) / (2 * step * volume)
            self.assertAlmostEqual(analytic_stress[voigt], numerical, delta=2e-7)

    def test_bond_across_periodic_boundary(self):
        atoms = Atoms("C2H2", positions=[[0.3, 5, 5], [9.1, 5, 5],
                                         [1.36, 5, 5], [8.04, 5, 5]],
                      cell=[10, 10, 10], pbc=True)
        model = (atoms.get_chemical_symbols(), ["C_1", "C_1", "H_", "H_"],
                 [dict(i=0, j=1, order="triple", image=[-1, 0, 0]),
                  dict(i=0, j=2, order="single", image=[0, 0, 0]),
                  dict(i=1, j=3, order="single", image=[0, 0, 0])])
        atoms.calc = self.calculator(atoms, model)
        force = atoms.get_forces()[0, 0]
        step = 1e-4
        atoms.positions[0, 0] += step
        plus = atoms.get_potential_energy()
        atoms.positions[0, 0] -= 2 * step
        minus = atoms.get_potential_energy()
        self.assertAlmostEqual(force, -(plus - minus) / (2 * step), delta=5e-5)

    def test_nonfinite_derivatives_have_a_clear_calculation_error(self):
        calculator = self.calculator(benzene())
        derivatives = self.directory / "invalid.drv"
        derivatives.write_text("energy NaN eV\ngradients cartesian eV/Ang 1\n1 *************** 0 0\n")
        with self.assertRaisesRegex(CalculationFailed, "nonfinite or overflowed"):
            calculator._read_derivatives(derivatives)

    def _optimizer(self, beta=90):
        atoms = benzene(radius=1.5)
        atoms.set_cell(Cell.fromcellpar([20, 20, 20, 90, beta, 90]), scale_atoms=True)
        source = self.directory / "benzene.cif"
        structure = Structure(atoms.cell.array, atoms.get_chemical_symbols(),
                              atoms.get_scaled_positions())
        CifWriter(structure, symprec=0.01, refine_struct=False).write_file(source)
        restored = read(source)
        calculator = self.calculator(restored)
        optimizer = GeometryOptimizer(source, calculator=calculator)
        return optimizer, check_symmetry(restored, symprec=0.01).number

    def test_optimization_preserves_ring_and_exported_spacegroup(self):
        for beta in (90, 95):
            with self.subTest(beta=beta):
                optimizer, original_group = self._optimizer(beta)
                destination = self.directory / f"optimized_{beta}.cif"
                atoms = optimizer.optimize(fmax=0.03, steps=500, output_path=destination)
                exported = read(destination)
                self.assertEqual(check_symmetry(exported, symprec=0.01).number, original_group)
                self.assertEqual(exported.info["spacegroup"].no, original_group)
                self.assertEqual(len(exported), 12)
                for i, j, order, image in atoms.calc.bonds:
                    if order == "resonant":
                        distance = atoms.get_distance(i, j, mic=True)
                        self.assertGreater(distance, 1.35)
                        self.assertLess(distance, 1.45)
                ring_positions = atoms.positions[np.array(atoms.get_chemical_symbols()) == "C"]
                singular_values = np.linalg.svd(ring_positions - ring_positions.mean(axis=0),
                                                compute_uv=False)
                self.assertLess(singular_values[-1], 1e-5)

    def test_nonconvergence_does_not_replace_previous_output(self):
        optimizer, _ = self._optimizer()
        destination = self.directory / "optimized.cif"
        destination.write_text("previous output")
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            optimizer.optimize(fmax=1e-12, steps=1, output_path=destination,
                               optimize_cell=False, logfile=self.directory / "optimization.log")
        self.assertEqual(destination.read_text(), "previous output")


if __name__ == "__main__":
    unittest.main()
