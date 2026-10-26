import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy
from openbabel import openbabel
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.spacegroup.symmetrize import check_symmetry
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


class addhydrogen:
    """给满占位、中性的有机周期模型补氢；键级根据几何推断，结果仍需检查。"""

    def __init__(self, structure=None, symprec=0.01, keep_symmetry=True):
        self.structure = structure
        self.symprec = symprec
        self.keep_symmetry = keep_symmetry

    def _build_molecule(self, structure):
        # 跨晶胞建立邻接表，不能把晶胞边界当成分子末端。
        i_list, j_list, shifts = neighbor_list(
            "ijS", structure, natural_cutoffs(structure, mult=1.2))
        neighbors = [[] for atom in structure]
        for i, j, shift in zip(i_list, j_list, shifts):
            vector = structure.positions[j] + shift @ structure.cell.array - structure.positions[i]
            if numpy.linalg.norm(vector) < 0.5:
                raise ValueError("存在距离小于 0.5 Å 的原子，请先检查重复原子或结构重叠。")
            neighbors[i].append((int(j), tuple(shift)))

        # 保留四层成键邻居，给 Open Babel 提供环和局部键级判断所需的环境。
        # 用展开的坐标处理跨边界键，也支持同一原子在不同晶胞中的多个邻居。
        nodes = {(i, (0, 0, 0)): i + 1 for i in range(len(structure))}
        frontier = list(nodes)
        for shell in range(4):
            following = []
            for i, image in frontier:
                for j, shift in neighbors[i]:
                    key = (j, tuple(numpy.array(image) + shift))
                    if key not in nodes:
                        nodes[key] = len(nodes) + 1
                        following.append(key)
            frontier = following

        molecule = openbabel.OBMol()
        molecule.BeginModify()
        for i, image in nodes:
            atom = molecule.NewAtom()
            atom.SetAtomicNum(int(structure.numbers[i]))
            position = structure.positions[i] + numpy.array(image) @ structure.cell.array
            atom.SetVector(*position)
        for (i, image), index in nodes.items():
            for j, shift in neighbors[i]:
                other = nodes.get((j, tuple(numpy.array(image) + shift)))
                if other is not None and index < other:
                    molecule.AddBond(index, other, 1)
        molecule.EndModify()
        molecule.PerceiveBondOrders()
        for atom in openbabel.OBMolAtomIter(molecule):
            openbabel.OBAtomAssignTypicalImplicitHydrogens(atom)

        # 短的多重键被识别为单键时不能继续补氢；原始示例还存在共线问题。
        for i in range(len(structure)):
            atom = molecule.GetAtom(i + 1)
            if (atom.GetAtomicNum() == 6 and atom.GetExplicitDegree() <= 3
                    and atom.GetExplicitValence() == atom.GetExplicitDegree()
                    and atom.GetImplicitHCount() >= 1):
                origin = numpy.array([atom.GetX(), atom.GetY(), atom.GetZ()])
                vectors = []
                for other in openbabel.OBAtomAtomIter(atom):
                    vector = numpy.array([other.GetX(), other.GetY(), other.GetZ()]) - origin
                    vectors.append(vector)
                    minimum = 0.9 * (covalent_radii[6] + covalent_radii[other.GetAtomicNum()])
                    if other.GetAtomicNum() in (6, 7, 8) and numpy.linalg.norm(vector) < minimum:
                        raise ValueError(f"第 {i + 1} 个碳的短键被识别为单键，键级判断不明确。"
                                         "请先修正初始几何，或使用带明确键级的化学模型。")
                if len(vectors) == 2:
                    cosine = numpy.dot(*vectors) / numpy.prod(numpy.linalg.norm(vectors, axis=1))
                    if cosine < -0.995:
                        raise ValueError(f"第 {i + 1} 个碳的两个邻居几乎共线，请先检查键级和初始几何。")
        return molecule

    def _keep_symmetry(self, structure, hydrogens, symmetry):
        # 只选一个等价母原子，再按输入结构的对称操作生成整组 H。
        scaled = structure.get_scaled_positions(wrap=False)
        cell = structure.cell.array
        inverse_cell = numpy.linalg.inv(cell)
        positions = []
        for group in numpy.unique(symmetry.equivalent_atoms):
            indices = numpy.flatnonzero(symmetry.equivalent_atoms == group)
            if len({len(hydrogens[i]) for i in indices}) != 1:
                raise ValueError("对称等价原子的补氢数量不同，自动键级判断不可靠。")
            representative = int(indices[0])
            children = {int(i): [] for i in indices}
            for position in hydrogens[representative]:
                h_scaled = position @ inverse_cell
                for rotation, translation in zip(symmetry.rotations, symmetry.translations):
                    parent = rotation @ scaled[representative] + translation
                    delta = parent - scaled[indices]
                    images = numpy.rint(delta)
                    distances = numpy.linalg.norm((delta - images) @ cell, axis=1)
                    match = int(numpy.argmin(distances))
                    if distances[match] > self.symprec:
                        raise ValueError("无法匹配对称等价的母原子。")
                    child = (rotation @ h_scaled + translation - images[match]) @ cell
                    values = children[int(indices[match])]
                    if not any(numpy.linalg.norm(child - old) < self.symprec for old in values):
                        values.append(child)
            for i in indices:
                if len(children[int(i)]) != len(hydrogens[i]):
                    raise ValueError("补氢方向与输入空间群不兼容。请检查键级和初始几何；"
                                     "若允许改变空间群，可设置 keep_symmetry=False。")
                positions.extend(children[int(i)])
        return positions

    def add_hydrogen(self):
        if self.structure is None:
            raise ValueError("请先传入 ASE Atoms，或者通过 run() 读取 CIF。")
        if not numpy.isfinite(self.symprec) or self.symprec <= 0:
            raise ValueError("symprec 必须大于 0。")
        structure = self.structure.copy()
        if not len(structure) or structure.cell.rank != 3 or not structure.pbc.all():
            raise ValueError("请输入具有完整晶胞和三维周期边界的结构。")
        if (not numpy.isfinite(structure.positions).all()
                or not numpy.isfinite(structure.cell.array).all()
                or abs(numpy.linalg.det(structure.cell.array)) < 1e-10):
            raise ValueError("原子坐标或晶胞包含无效数值。")
        for site in structure.info.get("occupancy", {}).values():
            if len(site) != 1 or not numpy.isclose(next(iter(site.values())), 1.0):
                raise ValueError("部分占位或混合占位模型需要先明确实际组成，不能直接补氢。")
        if set(structure.get_chemical_symbols()) - set("H B C N O F Si P S Cl Br I".split()):
            raise ValueError("当前脚本面向有机 COF；含金属等体系需要单独指定配位和价态。")
        symmetry = check_symmetry(structure, symprec=self.symprec)
        if symmetry is None:
            raise ValueError("无法识别输入结构的空间群。")
        declared = structure.info.get("spacegroup")
        if declared is not None and declared.no != symmetry.number:
            warnings.warn(f"CIF 标称空间群 {declared.no}，实际识别为 {symmetry.number}；"
                          "空间群检查以实际坐标为准。", stacklevel=2)

        molecule = self._build_molecule(structure)
        count = molecule.NumAtoms()
        expected_h = [molecule.GetAtom(i + 1).GetImplicitHCount() for i in range(len(structure))]
        molecule.AddHydrogens()
        hydrogens = [[] for atom in structure]
        for atom in openbabel.OBMolAtomIter(molecule):
            if atom.GetIdx() <= count:
                continue
            parent = next(openbabel.OBAtomAtomIter(atom)).GetIdx() - 1
            if parent < len(structure):
                hydrogens[parent].append(numpy.array([atom.GetX(), atom.GetY(), atom.GetZ()]))
        if [len(group) for group in hydrogens] != expected_h:
            raise ValueError("Open Babel 未能补齐预期的氢原子。")
        positions = (self._keep_symmetry(structure, hydrogens, symmetry) if self.keep_symmetry
                     else [position for group in hydrogens for position in group])
        if positions:
            if not numpy.isfinite(positions).all():
                raise ValueError("Open Babel 生成了无效的氢原子坐标。")
            structure += Atoms("H" * len(positions), positions=positions)
            i_list, j_list = neighbor_list("ij", structure, 0.6)
            if any(max(i, j) >= len(self.structure) for i, j in zip(i_list, j_list)):
                raise ValueError("新增氢原子与其他原子距离小于 0.6 Å，请检查补氢方向。")
        final = check_symmetry(structure, symprec=self.symprec)
        if final is None or (self.keep_symmetry and final.number != symmetry.number):
            raise ValueError("补氢后空间群发生变化，未更新结构或保存文件。")
        # 防止后续程序误用加氢前的 CIF 位点和占位元数据。
        structure.info = {key: value for key, value in structure.info.items()
                          if not key.startswith("_") and key not in ("occupancy", "spacegroup")}
        if "spacegroup_kinds" in structure.arrays:
            del structure.arrays["spacegroup_kinds"]
        self.structure = structure
        return structure

    def run(self, input_file, output_file):
        previous = self.structure
        self.structure = read(input_file)
        try:
            structure = self.add_hydrogen()
            output = Path(output_file)
            model = Structure(structure.cell.array, structure.get_chemical_symbols(),
                              structure.get_scaled_positions())
            # 先写临时文件，回读确认组分和空间群后再替换输出。
            with tempfile.NamedTemporaryFile(suffix=".cif", dir=output.parent, delete=False) as file:
                candidate = Path(file.name)
            try:
                CifWriter(model, symprec=self.symprec, angle_tolerance=-1,
                          refine_struct=False, significant_figures=12).write_file(candidate)
                restored = read(candidate)
                expected = check_symmetry(structure, symprec=self.symprec).number
                actual = check_symmetry(restored, symprec=self.symprec)
                if (actual is None or actual.number != expected
                        or Counter(restored.get_chemical_symbols()) != Counter(structure.get_chemical_symbols())):
                    raise ValueError("CIF 回读校验失败，未替换输出文件。")
                candidate.replace(output)
            finally:
                candidate.unlink(missing_ok=True)
        except Exception:
            self.structure = previous
            raise
        print(f"补氢完成：{structure.get_chemical_formula()}，空间群 {expected}，保存至 {output}")
        return structure


if __name__ == "__main__":
    # 用几何合理的模型演示；原始 example_10.cif 的连接会触发键级歧义检查。
    b = addhydrogen()
    b.run("Data/example_10_optimized.cif", "Data/example_10_hydrogen.cif")

