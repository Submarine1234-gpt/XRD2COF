import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy
from openbabel import openbabel
from ase.io import read
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.spacegroup.symmetrize import check_symmetry

from GeoOpti.addhydrogen import addhydrogen


class settingUFF:
    """用 Open Babel 为满占位、中性的有机周期模型分配 UFF 类型和键表。"""

    def __init__(self, structure=None, symprec=0.01, charges=None, uff_library=None):
        self.structure = structure
        self.symprec = symprec
        self.charges = charges
        self.uff_library = uff_library
        self.model = None

    def _get_bonds(self, structure, molecule, types):
        # 用相邻原子的真实坐标匹配原晶胞索引及 image，避免丢失跨晶胞连接。
        i_list, j_list, shifts = neighbor_list(
            "ijS", structure, natural_cutoffs(structure, mult=1.2))
        neighbors = [[] for atom in structure]
        for i, j, shift in zip(i_list, j_list, shifts):
            neighbors[i].append((int(j), shift))

        bonds = {}
        for i in range(len(structure)):
            atom = molecule.GetAtom(i + 1)
            for bond in openbabel.OBAtomBondIter(atom):
                other = bond.GetNbrAtom(atom)
                position = numpy.array([other.GetX(), other.GetY(), other.GetZ()])
                matches = [(j, image) for j, image in neighbors[i]
                           if numpy.linalg.norm(structure.positions[j]
                                                + image @ structure.cell.array - position) < 1e-7]
                if len(matches) != 1:
                    raise ValueError(f"无法把第 {i + 1} 个原子的邻居映射回周期晶胞。")
                j, image = matches[0]
                data = other.GetData("FFAtomType")
                if data is None or data.GetValue() != types[j]:
                    raise ValueError("同一原子的不同周期像得到不同 UFF 类型，请检查成键环境。")
                forward = (i, j, tuple(int(x) for x in image))
                reverse = (j, i, tuple(int(-x) for x in image))
                key = min(forward, reverse)
                if bond.IsAromatic():
                    order = "resonant"
                elif bond.IsAmide():
                    order = "amide"
                else:
                    order = {1: "single", 2: "double", 3: "triple"}.get(bond.GetBondOrder())
                if order is None:
                    raise ValueError("存在当前 GULP 接口不支持的键级。")
                if key in bonds and bonds[key] != order:
                    raise ValueError("同一周期键被识别成不同键级，请检查初始几何。")
                bonds[key] = order
        return [dict(i=i, j=j, order=order, image=list(image))
                for (i, j, image), order in sorted(bonds.items())]

    def setUFF(self):
        """返回带 uff_types 数组的 ASE Atoms；完整类型、键表和电荷保存在 self.model。"""
        if self.structure is None:
            raise ValueError("请传入 ASE Atoms，或者通过 run() 读取 CIF。")
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
                raise ValueError("部分占位或混合占位模型需要先明确实际组成。")
        if set(structure.get_chemical_symbols()) - set("H B C N O F Si P S Cl Br I".split()):
            raise ValueError("当前自动分型面向有机 COF；含金属等体系需要单独准备。")

        # 复用补氢脚本的四层周期邻居环境；中心晶胞原子排在最前面。
        # 此处只分型，缺氢时提示先补氢，不改变原子数量、坐标或晶胞。
        molecule = addhydrogen(structure, self.symprec)._build_molecule(structure)
        missing = [i + 1 for i in range(len(structure))
                   if molecule.GetAtom(i + 1).GetImplicitHCount()]
        if missing:
            raise ValueError(f"原子 {missing} 仍存在隐式氢，请先使用 addhydrogen 补氢。")
        forcefield = openbabel.OBForceField.FindForceField("UFF")
        if forcefield is None or not forcefield.Setup(molecule):
            raise ValueError("Open Babel 无法为该成键模型建立 UFF，请检查键级、价态和几何。")
        if not forcefield.GetAtomTypes(molecule):
            raise ValueError("Open Babel 未能导出 UFF 原子类型。")
        types = []
        for i in range(len(structure)):
            data = molecule.GetAtom(i + 1).GetData("FFAtomType")
            if data is None or not data.GetValue() or data.GetValue() == "Du":
                raise ValueError(f"第 {i + 1} 个原子没有有效的 UFF 类型。")
            types.append(data.GetValue())
        symmetry = check_symmetry(structure, symprec=self.symprec)
        if symmetry is None:
            raise ValueError("无法识别输入结构的空间群。")
        for group in numpy.unique(symmetry.equivalent_atoms):
            indices = numpy.flatnonzero(symmetry.equivalent_atoms == group)
            if len({types[i] for i in indices}) != 1:
                raise ValueError("对称等价原子的 UFF 类型不一致，请检查自动成键结果。")
        bonds = self._get_bonds(structure, molecule, types)
        degree = numpy.zeros(len(structure), dtype=int)
        for bond in bonds:
            degree[bond["i"]] += 1
            degree[bond["j"]] += 1
        if numpy.any(degree == 0) or any(t == "C_R" and degree[i] != 3 for i, t in enumerate(types)):
            raise ValueError("周期键表不完整，或芳香碳没有三个成键邻居。")

        # 电荷是单独的建模选择，默认沿用当前流程的零电荷方案。
        charges = (numpy.zeros(len(structure)) if self.charges is None
                   else numpy.asarray(self.charges, dtype=float))
        if (charges.shape != (len(structure),) or not numpy.isfinite(charges).all()
                or abs(charges.sum()) > 1e-6):
            raise ValueError("请为每个原子提供有限电荷，且周期模型的总电荷必须为零。")
        if self.uff_library is not None:
            library = Path(self.uff_library).read_text()
            for atom_type in set(types):
                if not re.search(r"^" + re.escape(atom_type) + r"\s+[-+\d.]", library, re.M):
                    raise ValueError(f"GULP 参数库中缺少类型 {atom_type}。")

        model = dict(symbols=structure.get_chemical_symbols(), types=types,
                     bonds=bonds, charges=charges.tolist())
        structure.set_array("uff_types", None)
        structure.set_array("uff_types", numpy.asarray(types))
        self.structure = structure
        self.model = model
        return structure

    def run(self, input_file=None, output_file=None):
        """读取 CIF 并生成同名 .uff.json；没有文件参数时仅处理构造函数传入的结构。"""
        if input_file is None:
            if output_file is not None:
                raise ValueError("保存 UFF JSON 时需要输入 CIF，以绑定文件和原子顺序。")
            return self.setUFF()
        source = Path(input_file)
        output = source.with_suffix(".uff.json") if output_file is None else Path(output_file)
        if source.resolve() == output.resolve():
            raise ValueError("UFF JSON 不能覆盖输入 CIF。")
        previous = self.structure, self.model
        try:
            self.structure = read(source)
            structure = self.setUFF()
            self.model["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            # 直接绑定读取的 CIF 和原子顺序，不重新写 CIF，以免索引发生变化。
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=output.parent,
                                             encoding="utf-8", delete=False) as file:
                candidate = Path(file.name)
            try:
                candidate.write_text(json.dumps(self.model, indent=2) + "\n", encoding="utf-8")
                candidate.replace(output)
            finally:
                candidate.unlink(missing_ok=True)
        except Exception:
            self.structure, self.model = previous
            raise
        print(f"UFF 类型：{dict(Counter(self.model['types']))}，周期键 {len(self.model['bonds'])} 条")
        print(f"已保存 {output}")
        return structure


if __name__ == "__main__":
    b = settingUFF()
    b.run("Data/example_10_hydrogen.cif")
