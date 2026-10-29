"""Compose an Isaac Sim scene: LiDAR collision mesh + Gaussian splat visual, in one frame.

  python make_scene_usd.py --collision collision/collision_mesh.usd --splat corridor_splat_gsframe.usdc \
      --T_gsplat_from_sim collision/T_gsplat_from_sim.txt --T_sim_from_ba collision/T_sim_from_ba.txt \
      --images data/joint4/final_txt/images.txt --out corridor_scene.usda

The splat USD (from omni.kit.converter.gsplat / usd_convert_gsplat) is in gsplat's normalized frame, so
it is placed under an Xform carrying inv(T_gsplat_from_sim) (a similarity, scale ~7.29) instead of baking
the transform into the data (which would also require rotating every SH coefficient).

Writes <out> with default prim /Corridor:
  /Corridor/Collision   reference to the collision mesh (invisible by default, still collides)
  /Corridor/Splat       Xform -> reference to the splat prim
and <out>.views.json with camera poses (eye/target, sim frame) sampled along both recordings.
"""
import argparse, json, os
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf

ap = argparse.ArgumentParser()
ap.add_argument("--collision", required=True); ap.add_argument("--splat", required=True)
ap.add_argument("--T_gsplat_from_sim", required=True); ap.add_argument("--T_sim_from_ba", required=True)
ap.add_argument("--images", help="COLMAP images.txt of the joint model, for camera views")
ap.add_argument("--out", required=True)
a = ap.parse_args()
out_dir = os.path.dirname(os.path.abspath(a.out))
rel = lambda p: os.path.relpath(os.path.abspath(p), out_dir)

T_sim_from_gs = np.linalg.inv(np.loadtxt(a.T_gsplat_from_sim))
src = Usd.Stage.Open(a.splat)
splat_root = (src.GetDefaultPrim() or src.GetPseudoRoot().GetChildren()[0]).GetPath()  # e.g. /corridor_splat

stage = Usd.Stage.CreateNew(a.out)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/Corridor"); stage.SetDefaultPrim(root.GetPrim())

col = stage.DefinePrim("/Corridor/Collision")
col.GetReferences().AddReference(rel(a.collision))          # collision USD default prim = /World
UsdGeom.Imageable(stage.GetPrimAtPath("/Corridor/Collision/Corridor/collision")).MakeInvisible()

splat = UsdGeom.Xform.Define(stage, "/Corridor/Splat")
# USD uses row vectors (p' = p * M), so the column-vector matrix is transposed
splat.AddTransformOp().Set(Gf.Matrix4d(T_sim_from_gs.T.tolist()))
# the reference goes on a typeless child: a local "Xform" type would override the referenced
# ParticleField3DGaussianSplat type and the splat would no longer be rendered
stage.DefinePrim("/Corridor/Splat/gaussians").GetReferences().AddReference(rel(a.splat), splat_root)
stage.GetRootLayer().Save()
print("wrote", a.out, "| splat scale", round(float(np.linalg.norm(T_sim_from_gs[:3, 0])), 4))

if a.images:
    T_sim_ba = np.loadtxt(a.T_sim_from_ba)
    views = {}
    lines = [l.split() for l in open(a.images) if not l.startswith("#")][::2]
    for s in lines:
        if len(s) < 10: continue
        qw, qx, qy, qz, tx, ty, tz = map(float, s[1:8])
        Rcw = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(qw, qx, qy, qz))).GetTranspose())  # world->cam (Gf is row-major)
        C = -Rcw.T @ [tx, ty, tz]; fwd = Rcw.T @ [0, 0, 1]                                # optical z axis
        eye = T_sim_ba[:3, :3] @ C + T_sim_ba[:3, 3]; d = T_sim_ba[:3, :3] @ fwd
        views.setdefault(s[9][:2], []).append({"image": s[9], "eye": eye.round(4).tolist(), "target": (eye + 2.0 * d).round(4).tolist()})
    views = {k: sorted(v, key=lambda x: x["image"])[::25] for k, v in views.items()}
    json.dump(views, open(a.out + ".views.json", "w"), indent=1)
    print("views:", {k: len(v) for k, v in views.items()})
