from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy
from ase import Atoms
from ase.build import molecule
from ase.io import read
from ase.spacegroup.symmetrize import check_symmetry

from GeoOpti.fakeforcite import GeometryOptimizer
from GeoOpti.gulp_uff import GulpUFF, load_uff_model
from GeoOpti.settingUFF import settingUFF


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Data/example_10_optimized.cif"
GULP = shutil.which(os.environ.get("GULP_EXECUTABLE", "gulp"))


def periodic_molecule(name):
    atoms = molecule(name)
    atoms.cell = [20, 21, 22]
    atoms.pbc = True
    atoms.positions += [0.2, 10, 11]
    atoms.wrap()
    return atoms


class SettingUFFTests(unittest.TestCase):
    def test_cof_types_bonds_and_unchanged_geometry(self):
        atoms = read(SOURCE)
        worker = settingUFF(atoms)
        result = worker.run()
        self.assertEqual(Counter(worker.model["types"]), dict(H_=8, C_R=12, C_2=4, N_2=4))
        self.assertEqual(Counter(b["order"] for b in worker.model["bonds"]),
                         dict(single=16, resonant=12, double=4))
        self.assertTrue(any(any(b["image"]) for b in worker.model["bonds"]))
        numpy.testing.assert_array_equal(result.positions, atoms.positions)
        numpy.testing.assert_array_equal(result.cell.array, atoms.cell.array)
        self.assertEqual(result.arrays["uff_types"].tolist(), worker.model["types"])
        self.assertEqual(check_symmetry(result, 0.01).number, 10)

    def test_molecular_types_and_benzene_across_boundary(self):
        for name, expected in [("C6H6", dict(C_R=6, H_=6)),
                               ("CH4", dict(C_3=1, H_=4)), ("H2O", dict(O_3=1, H_=2))]:
            with self.subTest(name=name):
                worker = settingUFF(periodic_molecule(name))
                worker.setUFF()
                self.assertEqual(Counter(worker.model["types"]), expected)
                if name == "C6H6":
                    self.assertEqual(len(worker.model["bonds"]), 12)
                    self.assertTrue(any(any(b["image"]) for b in worker.model["bonds"]))

    def test_pyridine_aromatic_nitrogen(self):
        angles = numpy.arange(6) * numpy.pi / 3
        ring = [[1.4 * numpy.cos(t), 1.4 * numpy.sin(t), 0] for t in angles]
        hydrogen = [[2.48 * numpy.cos(t), 2.48 * numpy.sin(t), 0] for t in angles[1:]]
        atoms = Atoms("NC5H5", positions=numpy.array(ring + hydrogen) + 10,
                      cell=[20, 21, 22], pbc=True)
        worker = settingUFF(atoms)
        worker.setUFF()
        self.assertEqual(Counter(worker.model["types"]), dict(N_R=1, C_R=5, H_=5))
        self.assertEqual(sum(b["order"] == "resonant" for b in worker.model["bonds"]), 6)

    def test_distinct_periodic_bonds_between_same_atom_indices(self):
        carbon = numpy.array([[0, 6, 6], [1.15, 6.7, 6]])
        hydrogen = []
        for i in range(2):
            vectors = numpy.array([carbon[1-i] + [offset * 2.45, 0, 0] - carbon[i]
                                   for offset in ([0, -1] if i == 0 else [0, 1])])
            direction = -(vectors / numpy.linalg.norm(vectors, axis=1)[:, None]).sum(axis=0)
            direction /= numpy.linalg.norm(direction)
            hydrogen.append(carbon[i] + 1.08 * direction)
        atoms = Atoms("C2H2", positions=numpy.vstack([carbon, hydrogen]),
                      cell=[2.45, 12, 13], pbc=True)
        worker = settingUFF(atoms)
        worker.setUFF()
        carbon_bonds = [b for b in worker.model["bonds"] if b["i"] == 0 and b["j"] == 1]
        self.assertEqual(len(carbon_bonds), 2)
        self.assertEqual({b["order"] for b in carbon_bonds}, {"single", "double"})
        self.assertEqual({tuple(b["image"]) for b in carbon_bonds}, {(0, 0, 0), (-1, 0, 0)})

    def test_json_binding_and_failed_export_preserves_previous_file(self):
        source_bytes = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.uff.json"
            worker = settingUFF()
            atoms = worker.run(SOURCE, output)
            model = load_uff_model(output, SOURCE, atoms)
            self.assertEqual(model["source_sha256"], hashlib.sha256(source_bytes).hexdigest())
            self.assertEqual(SOURCE.read_bytes(), source_bytes)
            saved = output.read_bytes()
            library = Path(directory) / "empty.lib"
            library.write_text("# no parameters\n")
            worker.uff_library = library
            with self.assertRaisesRegex(ValueError, "参数库"):
                worker.run(SOURCE, output)
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(worker.model, model)
            self.assertEqual(json.loads(output.read_text()), model)

    def test_missing_hydrogen_partial_occupancy_and_bad_geometry(self):
        atoms = read(SOURCE)
        with self.assertRaisesRegex(ValueError, "补氢"):
            settingUFF(atoms[atoms.numbers != 1]).setUFF()
        atoms.info["occupancy"] = {"0": {"C": 0.7}}
        with self.assertRaisesRegex(ValueError, "占位"):
            settingUFF(atoms).setUFF()
        with self.assertRaisesRegex(ValueError, "键级|共线"):
            settingUFF(read(ROOT / "Data/example_10.cif")).setUFF()

    def test_charge_choice_is_explicit(self):
        atoms = periodic_molecule("H2O")
        worker = settingUFF(atoms, charges=[-0.8, 0.4, 0.4])
        worker.setUFF()
        self.assertEqual(worker.model["charges"], [-0.8, 0.4, 0.4])
        with self.assertRaisesRegex(ValueError, "电荷"):
            settingUFF(atoms, charges=[1, 0, 0]).setUFF()


@unittest.skipUnless(GULP, "Set GULP_EXECUTABLE to the GULP binary")
class SettingUFFGulpTests(unittest.TestCase):
    def calculate(self, atoms):
        worker = settingUFF(atoms)
        worker.setUFF()
        model = worker.model
        calculator = GulpUFF(model["symbols"], model["types"], model["bonds"], model["charges"],
                             executable=GULP)
        try:
            atoms.calc = calculator
            return atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
        finally:
            calculator.close()

    def test_real_gulp_energy_and_atom_permutation(self):
        atoms = read(SOURCE)
        energy, forces, stress = self.calculate(atoms)
        # 同一优化模型及人工审查过的化学拓扑，已有参考能量。
        self.assertAlmostEqual(energy, 2.4591928378, places=7)
        self.assertTrue(numpy.isfinite(stress).all())
        order = numpy.random.default_rng(12).permutation(len(atoms))
        shuffled_energy, shuffled_forces, shuffled_stress = self.calculate(atoms[order])
        self.assertAlmostEqual(shuffled_energy, energy, places=7)
        numpy.testing.assert_allclose(shuffled_forces, forces[order], atol=2e-8, rtol=0)
        numpy.testing.assert_allclose(shuffled_stress, stress, atol=2e-9, rtol=0)

    def test_generated_model_runs_existing_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            model_file = directory / "model.uff.json"
            settingUFF().run(SOURCE, model_file)
            worker = GeometryOptimizer(SOURCE, uff_model=model_file, gulp_executable=GULP)
            try:
                result = worker.optimize(steps=20, output_path=directory / "optimized.cif")
            finally:
                worker.atoms.calc.close()
            self.assertEqual(check_symmetry(result, 0.01).number, 10)
            self.assertEqual(read(directory / "optimized.cif").get_chemical_formula(), "C16H8N4")


if __name__ == "__main__":
    unittest.main()
