"""Convert raw_0411_big_smash.zarr.zip -> demo_vitac2.rrd for Rerun web viewer.
Layout mirrors viz_rerun.py (task1 style).
"""
import sys, os
import numpy as np
import cv2
import trimesh
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation

_SRC = '/Users/zhangrongxuan/robot_visualization-2/src'
sys.path.insert(0, _SRC)
from imagecodecs_numcodecs import register_codecs
register_codecs()
from replay_buffer import ReplayBuffer

ZARR_PATH   = 'raw_0411_big_smash.zarr.zip'
OUTPUT_RRD  = 'demo_vitac2.rrd'
N_EPISODES  = 2
STRIDE      = 2
JPEG_Q      = 75
ROBOT_IDS   = [0, 1]
ROBOT_COLORS = {0: [220, 60, 60], 1: [60, 220, 60]}

# ── mesh loading (same as viz_rerun.py) ───────────────────────────────────────
def load_static_meshes():
    meshes = {}
    gripper_path = os.path.join(_SRC, "meshes", "夹爪.STL")
    if os.path.exists(gripper_path):
        base = trimesh.load(gripper_path)
        base.apply_scale(0.001)
        center = (base.bounds[0] + base.bounds[1]) / 2
        base.apply_translation(-center)
        rot = np.eye(4)
        rot[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
        base.apply_transform(rot)
        meshes["gripper_left"] = {
            "vertices": base.vertices.astype(np.float32),
            "faces":    base.faces.astype(np.int32),
            "normals":  base.vertex_normals.astype(np.float32),
        }
        right = base.copy()
        mirror = np.eye(4); mirror[1, 1] = -1
        right.apply_transform(mirror)
        meshes["gripper_right"] = {
            "vertices": right.vertices.astype(np.float32),
            "faces":    right.faces.astype(np.int32),
            "normals":  right.vertex_normals.astype(np.float32),
        }
    for side, fname in [
        ("left",  "Oculus_Meta_Quest_Touch_Plus_Controller_Left.stl"),
        ("right", "Oculus_Meta_Quest_Touch_Plus_Controller_Right.stl"),
    ]:
        fpath = os.path.join(_SRC, "meshes", fname)
        if os.path.exists(fpath):
            m = trimesh.load(fpath)
            m.apply_scale(0.0015)
            meshes[f"controller_{side}"] = {
                "vertices": m.vertices.astype(np.float32),
                "faces":    m.faces.astype(np.int32),
                "normals":  m.vertex_normals.astype(np.float32),
            }
    return meshes


# ── static scene (same as viz_rerun.py) ──────────────────────────────────────
def log_static_geometry(meshes):
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for r in ROBOT_IDS:
        rr.log(f"world/robot{r}/eef/axes",
               rr.Arrows3D(vectors=np.eye(3, dtype=np.float32) * 0.05,
                            origins=np.zeros((3, 3), dtype=np.float32),
                            colors=[[255,0,0],[0,255,0],[0,0,255]]),
               static=True)
        ctrl_key = f"controller_{'left' if r == 1 else 'right'}"
        if ctrl_key in meshes:
            m = meshes[ctrl_key]
            n = len(m["vertices"])
            ctrl_rot_quat = Rotation.from_euler("y", 90, degrees=True).as_quat()
            rr.log(f"world/robot{r}/eef/controller",
                   rr.Transform3D(translation=[0,0,0.05],
                                  quaternion=rr.Quaternion(xyzw=ctrl_rot_quat)),
                   static=True)
            rr.log(f"world/robot{r}/eef/controller/mesh",
                   rr.Mesh3D(vertex_positions=m["vertices"],
                              triangle_indices=m["faces"],
                              vertex_normals=m.get("normals"),
                              vertex_colors=np.full((n,3), [100,100,200], dtype=np.uint8)),
                   static=True)
        for side in ("left", "right"):
            key = f"gripper_{side}"
            if key in meshes:
                m = meshes[key]
                n = len(m["vertices"])
                rr.log(f"world/robot{r}/eef/gripper/{side}/mesh",
                       rr.Mesh3D(vertex_positions=m["vertices"],
                                  triangle_indices=m["faces"],
                                  vertex_normals=m.get("normals"),
                                  vertex_colors=np.full((n,3), [180,180,180], dtype=np.uint8)),
                       static=True)
            sensor_color = [0,220,0] if side == "left" else [220,0,0]
            rr.log(f"world/robot{r}/eef/gripper/{side}/sensor",
                   rr.Points3D(positions=[[0,0,0]], colors=[sensor_color], radii=[0.012]),
                   static=True)


# ── blueprint (same structure as viz_rerun.py) ────────────────────────────────
def make_blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(name="3D World", origin="world"),
                rrb.Horizontal(
                    rrb.TimeSeriesView(name="Robot 0 Position (m)", origin="timeseries/robot0"),
                    rrb.TimeSeriesView(name="Robot 1 Position (m)", origin="timeseries/robot1"),
                ),
                row_shares=[3, 2],
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="R0 Visual",  origin="cameras/robot0/visual"),
                    rrb.Spatial2DView(name="R0 L-Tact",  origin="cameras/robot0/left_tactile"),
                    rrb.Spatial2DView(name="R0 R-Tact",  origin="cameras/robot0/right_tactile"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="R1 Visual",  origin="cameras/robot1/visual"),
                    rrb.Spatial2DView(name="R1 L-Tact",  origin="cameras/robot1/left_tactile"),
                    rrb.Spatial2DView(name="R1 R-Tact",  origin="cameras/robot1/right_tactile"),
                ),
                rrb.TimeSeriesView(name="Gripper Width (m)", origin="timeseries"),
                row_shares=[2, 2, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.TopPanel(state=rrb.PanelState.Hidden),
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TimePanel(state=rrb.PanelState.Collapsed),
        collapse_panels=True,
    )


# ── main ──────────────────────────────────────────────────────────────────────
print(f"Loading {ZARR_PATH} ...")
rb = ReplayBuffer.create_from_path(ZARR_PATH, mode='r')
print(f"  {rb.n_episodes} episodes, {rb.episode_ends[-1]} total frames")
episode_starts = np.concatenate([[0], rb.episode_ends[:-1]])

blueprint = make_blueprint()
rr.init("vitac_big_smash", spawn=False)

print("Loading meshes ...")
meshes = load_static_meshes()
print(f"  gripper: {'ok' if 'gripper_left' in meshes else 'MISSING'}  "
      f"controllers: {'ok' if 'controller_left' in meshes else 'MISSING'}")
log_static_geometry(meshes)

trajectories = {r: [] for r in ROBOT_IDS}
global_t = 0

for ep_idx in range(min(N_EPISODES, rb.n_episodes)):
    start  = int(episode_starts[ep_idx])
    end    = int(rb.episode_ends[ep_idx])
    frames = range(start, end, STRIDE)
    print(f"  Episode {ep_idx}: {len(frames)} frames ...")

    # reset trajectories per episode
    trajectories = {r: [] for r in ROBOT_IDS}

    for frame_idx in frames:
        rr.set_time("frame", sequence=global_t)

        for r in ROBOT_IDS:
            color = ROBOT_COLORS[r]

            # ── eef pose ──────────────────────────────────────────────────────
            pos    = rb.data[f'robot{r}_eef_pos'][frame_idx].astype(np.float32)
            rot_aa = rb.data[f'robot{r}_eef_rot_axis_angle'][frame_idx]
            rot_mat, _ = cv2.Rodrigues(rot_aa)
            quat = Rotation.from_matrix(rot_mat).as_quat()  # xyzw

            trajectories[r].append(pos.copy())
            rr.log(f"world/robot{r}/eef",
                   rr.Transform3D(translation=pos,
                                  quaternion=rr.Quaternion(xyzw=quat)))

            traj = np.array(trajectories[r], dtype=np.float32)
            rr.log(f"world/robot{r}/trajectory",
                   rr.LineStrips3D([traj], colors=[color], radii=[0.003]))

            rr.log(f"timeseries/robot{r}/eef_x", rr.Scalars(float(pos[0])))
            rr.log(f"timeseries/robot{r}/eef_y", rr.Scalars(float(pos[1])))
            rr.log(f"timeseries/robot{r}/eef_z", rr.Scalars(float(pos[2])))

            # ── gripper ───────────────────────────────────────────────────────
            grip = float(rb.data[f'robot{r}_gripper_width'][frame_idx, 0])
            rr.log(f"timeseries/robot{r}/gripper_width", rr.Scalars(grip))
            offset = max(grip * 0.5, 0.03)
            for side, sign in [("left", -1), ("right", 1)]:
                rr.log(f"world/robot{r}/eef/gripper/{side}",
                       rr.Transform3D(translation=[0.02, sign * offset, -0.04]))

            # ── cameras ───────────────────────────────────────────────────────
            for key, cam_path in [
                (f'camera{r}_rgb',           f"cameras/robot{r}/visual"),
                (f'camera{r}_left_tactile',  f"cameras/robot{r}/left_tactile"),
                (f'camera{r}_right_tactile', f"cameras/robot{r}/right_tactile"),
            ]:
                img = rb.data[key][frame_idx]
                _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                rr.log(cam_path, rr.EncodedImage(contents=bytes(buf),
                                                  media_type="image/jpeg"))

        global_t += 1

print(f"Saving {OUTPUT_RRD} ...")
rr.save(OUTPUT_RRD, default_blueprint=blueprint)
print(f"Done. {global_t} frames logged.")
