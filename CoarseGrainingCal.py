import pymatgen

class CoarseGrainingCal :
    def __init__ (self,cif_path,tensor):
        self.cif_path = cif_path
        self.tensor = tensor

    def load_file(self):
        structure = pymatgen.core.structure.Structure.from_file(self.cif_path)
        return structure

    def coarse_graining(self):
        pass

    def run(self):
        self.load_file()
        self.coarse_graining()
        return self.tensor
        



if __name__ == "__main__":
    print("a")