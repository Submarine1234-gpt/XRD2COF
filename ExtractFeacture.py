import numpy
import pymatgen.core.structure
import os
import re

class ExtractFeacture:
    def __init__(self, directory: str):
        self.directory = directory
        self.structures = []

    def natural_sort_key(self, filename):
        """自然排序: 1.cif, 2.cif, 10.cif 而不是 1.cif, 10.cif, 2.cif"""
        return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]

    def load_file(self):
        cif_files = [f for f in os.listdir(self.directory) if f.endswith(".cif")]
        cif_files.sort(key=self.natural_sort_key)
        for file in cif_files:
            filepath = os.path.join(self.directory, file)
            structure = pymatgen.core.structure.Structure.from_file(filepath)
            self.structures.append(structure)
        print(f"Loaded {len(self.structures)} CIF files")
        return self.structures

    def get_topology(self):
        pass

    def combine_features(self):
        tensor = numpy.concatenate((struc_tensor, xrd_tensor), axis=1)
        pass


if __name__ == "__main__":
    ef = ExtractFeacture("all_cifs")
    ef.load_file()
    print(f"Total structures: {len(ef.structures)}")