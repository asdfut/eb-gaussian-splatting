"""Load the EB corridor in Isaac Lab: Gaussian-splat visual + LiDAR collision mesh, drop rigid bodies
on the floor and (optionally) render camera views with the ZED intrinsics.

  python load_corridor_scene.py --viz kit                     # interactive window (Isaac Lab 3.x)
  python load_corridor_scene.py --save_dir renders/          # headless (default): renders + collision check
  options: --show_collision (draw the collision mesh), --num_objects N, --steps N, --render_frames N

The scene is in metres, Z up, floor at z = 0 (see collision/T_gsplat_from_sim.txt for the frame).
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="EB corridor: 3DGS visual + collision mesh in Isaac Lab.")
parser.add_argument("--scene", default=os.path.join(HERE, "corridor_scene.usda"))
parser.add_argument("--show_collision", action="store_true", help="make the collision mesh visible")
parser.add_argument("--num_objects", type=int, default=10, help="balls dropped along the robot path")
parser.add_argument("--steps", type=int, default=480, help="physics steps (dt = 1/120 s)")
parser.add_argument("--save_dir", default=None, help="save RGB renders from recorded camera poses here")
parser.add_argument("--num_views", type=int, default=4, help="camera views to render per recording")
parser.add_argument("--render_frames", type=int, default=64, help="RTX accumulation frames per view (more = less noise)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.save_dir:
    args.enable_cameras = True
# Gaussian splats (UsdVol ParticleField) are drawn by RTX Real-Time 2.0 and need multi-GPU rendering off
args.kit_args = " ".join(filter(None, [getattr(args, "kit_args", "") or "",
    "--/rtx/rendermode=RealTimePathTracing", "--/rtx/modes/rt2/enabled=true", "--/renderer/multiGpu/enabled=false"]))
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from pxr import UsdGeom

BALL_RADIUS = 0.1
# ZED left camera (rectified 1280x720, fx = 528 px) -> pinhole with the same horizontal FOV
IMG_W, IMG_H, FX = 1280, 720, 528.05
APERTURE = 20.955


def design_scene(views: dict) -> tuple[RigidObject, Camera | None]:
    scene_cfg = sim_utils.UsdFileCfg(usd_path=os.path.abspath(args.scene))
    scene_cfg.func("/World/Corridor", scene_cfg)
    if args.show_collision:
        stage = sim_utils.get_current_stage()
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Corridor/Collision/Corridor/collision")).MakeVisible()
    light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/Light", light_cfg)

    # balls 1 m above the floor along the h1 robot path (known free space), halfway between recorded
    # camera positions so that no rendered camera ends up inside a ball
    path = np.array([v["eye"] for v in views["h1"]])
    mid = 0.5 * (path[:-1] + path[1:])
    for i, j in enumerate(np.linspace(0, len(mid) - 1, args.num_objects).astype(int)):
        sim_utils.create_prim(f"/World/Drop_{i:02d}", "Xform", translation=(mid[j][0], mid[j][1], 1.0))
    balls = RigidObject(RigidObjectCfg(
        prim_path="/World/Drop_.*/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.1)),
        ),
    ))

    camera = None
    if args.save_dir:
        sim_utils.create_prim("/World/CamRig", "Xform")
        camera = Camera(CameraCfg(
            prim_path="/World/CamRig/Camera",
            update_period=0, height=IMG_H, width=IMG_W, data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=APERTURE * FX / IMG_W, horizontal_aperture=APERTURE, clipping_range=(0.05, 100.0)
            ),
        ))
    return balls, camera


def main():
    views = json.load(open(args.scene + ".views.json"))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    balls, camera = design_scene(views)
    eye0 = views["h3"][0]
    sim.set_camera_view(eye0["eye"], eye0["target"])
    sim.reset()
    import carb
    stage = sim_utils.get_current_stage()
    splats = [p for p in stage.Traverse() if p.GetTypeName().startswith("ParticleField")]
    n = len(splats[0].GetAttribute("positions").Get()) if splats else 0
    print(f"[corridor] gaussian splat prims: {[str(p.GetPath()) for p in splats]} ({n} splats) | "
          f"rtx/rendermode = {carb.settings.get_settings().get('/rtx/rendermode')}")
    print("[corridor] scene loaded, simulating", args.steps, "steps")

    for _ in range(args.steps):
        sim.step()
        balls.update(sim.get_physics_dt())
    z = balls.data.root_pos_w[:, 2].cpu().numpy()
    on_floor = np.abs(z - BALL_RADIUS) < 0.05
    print(f"[corridor] ball heights after {args.steps / 120:.1f} s: {np.round(z, 3).tolist()}")
    print(f"[corridor] collision check: {on_floor.sum()}/{len(z)} balls resting on the floor (z ~ {BALL_RADIUS} m)")

    if camera is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        import imageio.v2 as imageio
        for rec in ("h1", "h3"):
            for k in np.linspace(0, len(views[rec]) - 1, args.num_views).astype(int):
                v = views[rec][k]
                camera.set_world_poses_from_view(torch.tensor([v["eye"]], device=sim.device),
                                                 torch.tensor([v["target"]], device=sim.device))
                for _ in range(args.render_frames):  # path-traced gaussians need many subframes
                    sim.render()
                camera.update(dt=0.0)
                rgb = camera.data.output["rgb"][0].cpu().numpy()
                name = v["image"].replace("/", "_").replace(".jpg", ".png")
                imageio.imwrite(os.path.join(args.save_dir, name), rgb)
                print("[corridor] saved", name, "mean rgb", rgb.reshape(-1, rgb.shape[-1]).mean(0).round(1).tolist())

    if app_launcher.has_window:
        while simulation_app.is_running():
            sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
