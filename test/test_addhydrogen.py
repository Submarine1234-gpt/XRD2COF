import tempfile
import unittest
import warnings
from pathlib import Path

import numpy
from ase import Atoms
from ase.build import molecule
from ase.io import read, write
from ase.spacegroup.symmetrize import check_symmetry

from GeoOpti.addhydrogen import addhydrogen


ROOT = Path(__file__).resolve().parents[1]


class TestAddHydrogen(unittest.TestCase):
    def test_complete_structure_not_modified(self):
        atoms = read(ROOT / "Data/example_10_optimized.cif")
        result = addhydrogen(atoms).add_hydrogen()
        self.assertEqual(len(result), len(atoms))
        numpy.testing.assert_array_equal(result.positions, atoms.positions)

    def test_periodic_cof_and_repeat(self):
        original = read(ROOT / "Data/example_10_optimized.cif")
        heavy = original[original.numbers != 1]
        heavy.info.clear()
        result = addhydrogen(heavy).add_hydrogen()
        self.assertEqual(result.get_chemical_formula(), "C16H8N4")
        self.assertEqual(check_symmetry(result, symprec=0.01).number, 10)
        numpy.testing.assert_array_equal(result.cell.array, heavy.cell.array)
        numpy.testing.assert_array_equal(result.positions[:len(heavy)], heavy.positions)
        self.assertEqual(len(addhydrogen(result).add_hydrogen()), len(result))
        # 每个新增 H 都只有一个合理长度的近邻键。
        for i in range(len(heavy), len(result)):
            distances = result.get_distances(i, range(len(heavy)), mic=True)
            self.assertEqual(numpy.count_nonzero(distances < 1.25), 1)
            self.assertGreater(distances.min(), 0.95)

    def test_benzene_crossing_cell_boundary(self):
        heavy = molecule("C6H6")
        heavy = heavy[heavy.numbers != 1]
        heavy.cell = [12, 13, 14]
        heavy.pbc = True
        heavy.positions += [0.2, 6, 7]
        heavy.wrap()
        result = addhydrogen(heavy).add_hydrogen()
        self.assertEqual(result.get_chemical_formula(), "C6H6")
        self.assertEqual(check_symmetry(result, 0.01).number,
                         check_symmetry(heavy, 0.01).number)
        numpy.testing.assert_array_equal(result.positions[:6], heavy.positions)

    def test_ambiguous_geometry_does_not_overwrite(self):
        for source in ["example_10.cif", "example_10_prepared.cif"]:
            heavy = read(ROOT / "Data" / source)
            heavy = heavy[heavy.numbers != 1]
            heavy.info.clear()
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                input_file = Path(directory) / "input.cif"
                output_file = Path(directory) / "output.cif"
                write(input_file, heavy)
                output_file.write_text("keep this file")
                worker = addhydrogen()
                with warnings.catch_warnings(), self.assertRaisesRegex(ValueError, "键级"):
                    warnings.simplefilter("ignore")
                    worker.run(input_file, output_file)
                self.assertEqual(output_file.read_text(), "keep this file")
                self.assertIsNone(worker.structure)

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            atoms = read(ROOT / "Data/example_10_optimized.cif")
            heavy = atoms[atoms.numbers != 1]
            heavy.info.clear()
            input_file = Path(directory) / "input.cif"
            result_file = Path(directory) / "hydrogen.cif"
            write(input_file, heavy)
            with self.assertWarnsRegex(UserWarning, "标称空间群"):
                result = addhydrogen().run(input_file, result_file)
            restored = read(result_file)
            self.assertEqual(result.get_chemical_formula(), "C16H8N4")
            self.assertEqual(restored.get_chemical_formula(), "C16H8N4")
            self.assertEqual(restored.info["spacegroup"].no, 10)

    def test_partial_occupancy_rejected(self):
        atoms = read(ROOT / "Data/example_10_prepared.cif")
        atoms.info["occupancy"] = {"0": {"C": 0.7}}
        with self.assertRaisesRegex(ValueError, "占位"):
            addhydrogen(atoms).add_hydrogen()

    def test_incompatible_symmetry_requires_explicit_opt_out(self):
        # 单个 C 的立方对称性无法由四面体 CH4 保持。
        carbon = Atoms("C", positions=[[5, 5, 5]], cell=[10, 10, 10], pbc=True)
        worker = addhydrogen(carbon)
        with self.assertRaisesRegex(ValueError, "空间群"):
            worker.add_hydrogen()
        self.assertIs(worker.structure, carbon)
        result = addhydrogen(carbon, keep_symmetry=False).add_hydrogen()
        self.assertEqual(result.get_chemical_formula(), "CH4")


if __name__ == "__main__":
    unittest.main()
