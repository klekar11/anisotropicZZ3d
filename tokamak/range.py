import meshio
import numpy as np

m = meshio.read("/home/karapas/Desktop/fenicsx-tutos/tokamak/TCV.mesh")
coords = m.points

print("Bounding box:")
for i, axis in enumerate(["X", "Y", "Z"]):
    lo, hi = coords[:, i].min(), coords[:, i].max()
    print(f"  {axis}: [{lo:.4f}, {hi:.4f}]  range = {hi - lo:.4f}")

print(f"\nTotal vertices: {len(coords)}")
print(f"Total tets: {len(m.cells_dict.get('tetra', []))}")