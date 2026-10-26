"""Periodic GULP UFF single points for ASE, using an explicit chemical model.

GULP does not infer reliable aromatic types or bond orders from a generic CIF.
The caller must supply types and periodic bonds for the expanded ASE atom order.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from ase.calculators.calculator import Calculator, CalculationFailed, all_changes


def load_uff_model(path, structure_path, atoms):
    """Bind a reviewed topology to the exact source CIF and expanded atom order."""
    model = json.loads(Path(path).read_text())
    digest = hashlib.sha256(Path(structure_path).read_bytes()).hexdigest()
    if model.get("source_sha256") != digest:
        raise ValueError("UFF model source_sha256 does not match the input CIF.")
    if model.get("symbols") != atoms.get_chemical_symbols():
        raise ValueError("UFF model symbols must match the expanded ASE atom order.")
    return model


class GulpUFF(Calculator):
    """Evaluate a fixed UFF topology; ASE controls positions and cell changes.

    Bond records use zero-based ``i``, ``j`` and integer ``image``. The vector is
    positions[j] + image @ cell - positions[i]. No shell command or GULP internal
    optimization is used. Energies, forces and stress are returned in ASE units.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, symbols, atom_types, bonds, charges=None,
                 executable="gulp", library=None, directory=None, **kwargs):
        self._scratch = tempfile.TemporaryDirectory(prefix="xrd2cof-gulp-") if directory is None else None
        directory = self._scratch.name if self._scratch is not None else directory
        super().__init__(directory=directory, **kwargs)
        self.symbols = list(symbols)
        self.uff_types = list(atom_types)
        if len(self.uff_types) != len(self.symbols):
            raise ValueError("One UFF atom type is required per atom.")
        for symbol, uff_type in zip(self.symbols, self.uff_types):
            match = re.fullmatch(r"([A-Z][a-z]?)[A-Za-z0-9_+]*", uff_type)
            if match is None or match.group(1) != symbol:
                raise ValueError(f"Invalid UFF type {uff_type!r} for {symbol}.")

        self.charges = np.zeros(len(self.symbols)) if charges is None else np.array(charges, dtype=float)
        if self.charges.shape != (len(self.symbols),) or not np.all(np.isfinite(self.charges)):
            raise ValueError("Provide one finite charge per atom.")
        if abs(self.charges.sum()) > 1e-6:
            raise ValueError("The periodic UFF model must be charge neutral.")

        self.bonds = []
        seen = set()
        neighbors = [set() for _ in self.symbols]
        allowed_orders = {"single", "double", "triple", "resonant", "amide"}
        for bond in bonds:
            i, j = bond["i"], bond["j"]
            image = tuple(bond.get("image", [0, 0, 0]))
            order = bond["order"]
            if (type(i) is not int or type(j) is not int
                    or not 0 <= i < len(self.symbols) or not 0 <= j < len(self.symbols)
                    or len(image) != 3 or any(type(k) is not int for k in image)
                    or order not in allowed_orders or (i == j and image == (0, 0, 0))):
                raise ValueError(f"Invalid periodic bond: {bond}")
            key = min((i, j, image), (j, i, tuple(-k for k in image)))
            if key in seen:
                raise ValueError(f"Duplicate periodic bond: {bond}")
            seen.add(key)
            neighbors[i].add((j, image))
            neighbors[j].add((i, tuple(-k for k in image)))
            self.bonds.append((i, j, order, image))
        if any(not adjacent for adjacent in neighbors):
            raise ValueError("Every atom in this molecular UFF model needs explicit bonds.")
        for i, uff_type in enumerate(self.uff_types):
            if uff_type == "C_R" and len(neighbors[i]) != 3:
                raise ValueError(
                    f"Aromatic carbon {i} has {len(neighbors[i])} neighbors; "
                    "provide its complete bonding, including H where required."
                )

        executable = os.fspath(executable)
        self.executable = shutil.which(executable)
        if self.executable is None:
            raise FileNotFoundError(f"GULP executable not found: {executable}")
        if library is None:
            lib_dir = os.environ.get("GULP_LIB")
            if lib_dir is None:
                candidate = Path(self.executable).resolve().parent.parent / "Libraries"
                lib_dir = os.fspath(candidate)
            library = Path(lib_dir) / "uff.lib"
        self.library = Path(library).expanduser().resolve()
        if not self.library.is_file():
            raise FileNotFoundError(f"UFF library not found: {self.library}")
        if any(char.isspace() for char in str(self.library)):
            raise ValueError("Use a GULP library path without whitespace.")
        # Do not let an unknown label silently produce zero interaction energy.
        library_text = self.library.read_text()
        for uff_type in set(self.uff_types):
            if not re.search(r"^" + re.escape(uff_type) + r"\s+[-+\d.]", library_text, re.M):
                raise ValueError(f"UFF parameters missing for {uff_type} in {self.library}")

    def close(self):
        scratch = getattr(self, "_scratch", None)
        if scratch is not None:
            scratch.cleanup()
            self._scratch = None

    def __del__(self):
        self.close()

    def _input(self, atoms):
        if atoms.get_chemical_symbols() != self.symbols:
            raise ValueError("UFF topology no longer matches the atom order.")
        if not np.all(atoms.pbc) or atoms.cell.volume <= 0:
            raise ValueError("This calculator requires a three-dimensional periodic cell.")
        aliases = {}
        for symbol, uff_type in zip(self.symbols, self.uff_types):
            aliases.setdefault(uff_type, f"{symbol}{len(aliases) + 1}")
        lines = ["gradient conp stress_out noautobond nosymmetry norotate nowrap",
                 "vectors"]
        lines += [" ".join(f"{x:.14f}" for x in row) for row in atoms.cell.array]
        lines.append("cartesian")
        for xyz, uff_type, charge in zip(atoms.positions, self.uff_types, self.charges):
            lines.append(f"{aliases[uff_type]} core "
                         + " ".join(f"{x:.14f}" for x in xyz) + f" {charge:.12f}")
        lines += ["species"]
        lines += [f"{alias} core {uff_type}" for uff_type, alias in aliases.items()]
        lines += [f"library {self.library}"]
        for i, j, order, image in self.bonds:
            lines.append(f"connect {i + 1} {j + 1} {order} " + " ".join(map(str, image)))
        lines += ["fix_atom none", "output drv result"]
        return "\n".join(lines) + "\n"

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        text = self._input(self.atoms)
        (directory / "singlepoint.gin").write_text(text)
        drv_path = directory / "result.drv"
        drv_path.unlink(missing_ok=True)
        try:
            process = subprocess.run([self.executable], input=text, text=True,
                                     capture_output=True, cwd=directory, timeout=120)
        except subprocess.TimeoutExpired as error:
            raise CalculationFailed("GULP single point exceeded 120 seconds.") from error
        output = process.stdout + process.stderr
        (directory / "singlepoint.got").write_text(output)
        if (process.returncode != 0 or "Job Finished" not in output
                or "Universal Force Field (UFF) species output" not in output
                or not drv_path.exists()):
            raise CalculationFailed(f"GULP UFF failed; inspect {directory / 'singlepoint.got'}\n{output[-1500:]}")
        self.results = self._read_derivatives(drv_path)

    def _read_derivatives(self, path):
        text = path.read_text()
        if re.search(r"\*{3,}|\b(?:nan|[+-]?inf(?:inity)?)\b", text, re.I):
            raise CalculationFailed(
                "GULP returned nonfinite or overflowed derivatives. Check the input "
                "geometry for collinear trigonal sites or overlapping atoms; "
                f"inspect {Path(self.directory) / 'singlepoint.got'}."
            )
        lines = text.splitlines()
        energy = forces = strain = None
        coordinates = numbers = None
        for n, line in enumerate(lines):
            if line.startswith("energy "):
                energy = float(line.split()[1])
            elif line.startswith("coordinates cartesian Angstroms"):
                count = int(line.split()[-1])
                rows = [row.split() for row in lines[n + 1:n + 1 + count]]
                numbers = [int(row[1]) for row in rows]
                coordinates = np.array([[float(x) for x in row[2:5]] for row in rows])
            elif line.startswith("gradients cartesian eV/Ang"):
                count = int(line.split()[-1])
                rows = [row.split() for row in lines[n + 1:n + 1 + count]]
                if [int(row[0]) for row in rows] != list(range(1, len(self.symbols) + 1)):
                    raise CalculationFailed("GULP derivative atom indices changed.")
                forces = -np.array([[float(x) for x in row[1:4]] for row in rows])
            elif line.startswith("gradients strain eV"):
                strain = np.array([float(x) for row in lines[n + 1:n + 3] for x in row.split()])
        if energy is None or forces is None or strain is None or coordinates is None:
            raise CalculationFailed("GULP did not return complete energy, forces and strain derivatives.")
        if (numbers != self.atoms.numbers.tolist()
                or coordinates.shape != self.atoms.positions.shape
                or not np.allclose(coordinates, self.atoms.positions, atol=1e-7, rtol=0)):
            raise CalculationFailed("GULP reordered, wrapped or rotated the input structure.")
        if forces.shape != (len(self.symbols), 3) or strain.shape != (6,):
            raise CalculationFailed("Unexpected GULP derivative dimensions.")
        # GULP's strain gradient is in eV; division by volume gives ASE's
        # eV/Angstrom**3 stress, with Voigt order xx, yy, zz, yz, xz, xy.
        stress = strain / self.atoms.get_volume()
        if not (np.isfinite(energy) and np.all(np.isfinite(forces)) and np.all(np.isfinite(stress))):
            raise CalculationFailed("Nonfinite GULP results.")
        return dict(energy=energy, free_energy=energy, forces=forces, stress=stress)
