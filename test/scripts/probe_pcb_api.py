import inspect, traceback
import pycofbuilder as pcb

def sig(obj, name):
    fn = getattr(obj, name, None)
    if fn is None:
        return f"{name}: MISSING"
    try:
        return f"{name}{inspect.signature(fn)}"
    except (TypeError, ValueError):
        return f"{name}: <unavailable>"

print("version:", getattr(pcb, "__version__", "?"))
print("file   :", getattr(pcb, "__file__", "?"))

fw = pcb.Framework(verbosity="none")
print("\n=== Framework ===")
for n in ("from_building_blocks", "from_name", "create_hcb_a_structure",
          "create_sql_a_structure", "create_fxt_a_structure",
          "create_kgd_structure", "create_hxl_a_structure", "to_ase", "save"):
    print(" ", sig(fw, n))
for a in ("available_2D_topologies", "available_stacking",
          "available_topologies", "symm_tol", "angle_tol", "dist_threshold"):
    if hasattr(fw, a):
        print(f"  {a} = {getattr(fw, a)}")

print("\n=== BuildingBlock ===")
print("  __init__", sig(pcb.BuildingBlock, "__init__"))
for n in ("from_file", "from_name", "get_X_points", "get_buildingblock_list",
          "replace_X", "save", "to_ase"):
    print(" ", sig(pcb.BuildingBlock, n))

print("\n=== 冒烟 ===")
bb_t3 = None
try:
    bb_t3 = pcb.BuildingBlock(name="T3_BENZ_NH2_OH", conector="NH2")
    print("  T3 n_atoms =", bb_t3.n_atoms)
    pts = bb_t3.get_X_points()
    print("  get_X_points() type =", type(pts).__name__, "len =", len(pts))
    if isinstance(pts, tuple):
        print("    tuple 元素类型 =", [type(p).__name__ for p in pts])
except Exception:
    traceback.print_exc()

try:
    bb_l2 = pcb.BuildingBlock(name="L2_BENZ_CHO_H", conector="CHO")
    cof = pcb.Framework(verbosity="none")
    cof.from_building_blocks(bb_t3, bb_l2, "HCB_A", "AA")
    print("  built: atoms =", cof.n_atoms, "sg =", cof.space_group)
    print("  lattice =\n", cof.lattice)
    print("  atom_pos[:2] =", cof.atom_pos[:2])
    print("  atom_types[:4] =", cof.atom_types[:4])
    print("  to_ase ok, n =", len(cof.to_ase()))
except Exception:
    traceback.print_exc()
