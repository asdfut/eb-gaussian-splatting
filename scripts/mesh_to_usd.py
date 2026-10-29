"""Convert the collision OBJ into a USD static collider for Isaac Sim / Isaac Lab.

Run with an interpreter that has `pxr` (e.g. the isaaclab conda env):
  python mesh_to_usd.py collision_mesh.obj collision_mesh.usd
Stage: Z-up, metres. Prim /World/Corridor/collision is a triangle mesh with CollisionAPI
(approximation "none" = exact triangle mesh, valid for static colliders).
"""
import sys
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Vt, Gf, Sdf

src, dst = sys.argv[1], sys.argv[2]
V, F = [], []
for line in open(src):
    if line.startswith("v "):
        V.append([float(x) for x in line.split()[1:4]])
    elif line.startswith("f "):
        F.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
V, F = np.asarray(V, np.float32), np.asarray(F, np.int32)

stage = Usd.Stage.CreateNew(dst)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
UsdGeom.Xform.Define(stage, "/World/Corridor")
mesh = UsdGeom.Mesh.Define(stage, "/World/Corridor/collision")
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel()))
mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
mesh.CreateDoubleSidedAttr(True)
lo, hi = V.min(0), V.max(0)
mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
mesh.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.6, 0.65)])
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
stage.GetRootLayer().Save()
print(f"wrote {dst}: {len(V)} vertices, {len(F)} triangles, extent {lo.round(2)} .. {hi.round(2)}")
