"""Convert raw_0411_big_smash.zarr.zip -> demo_vitac2.rrd for Rerun web viewer."""
import sys
import numpy as np
import cv2
import rerun as rr

sys.path.insert(0, '/Users/zhangrongxuan/robot_visualization-2/src')
from imagecodecs_numcodecs import register_codecs
register_codecs()
from replay_buffer import ReplayBuffer

ZARR_PATH = 'raw_0411_big_smash.zarr.zip'
OUTPUT_RRD = 'demo_vitac2.rrd'
N_EPISODES = 2   # number of episodes to export
STRIDE = 2       # frame stride (keep every Nth frame)
IMG_SIZE = 224   # keep original 224x224
JPEG_Q = 75      # JPEG quality

print(f"Loading {ZARR_PATH} ...")
rb = ReplayBuffer.create_from_path(ZARR_PATH, mode='r')
print(f"  {rb.n_episodes} episodes, {rb.episode_ends[-1]} total frames")

episode_starts = np.concatenate([[0], rb.episode_ends[:-1]])

rr.init("vitac_big_smash", spawn=False)

global_t = 0
for ep_idx in range(min(N_EPISODES, rb.n_episodes)):
    start = int(episode_starts[ep_idx])
    end   = int(rb.episode_ends[ep_idx])
    frames = range(start, end, STRIDE)
    print(f"  Episode {ep_idx}: frames {start}-{end}, exporting {len(frames)} frames ...")

    for frame_idx in frames:
        rr.set_time("frame", sequence=global_t)

        # --- cameras & tactile ---
        for cam_id in range(2):
            rgb = rb.data[f'camera{cam_id}_rgb'][frame_idx]          # (224,224,3) uint8
            lt  = rb.data[f'camera{cam_id}_left_tactile'][frame_idx]
            rt  = rb.data[f'camera{cam_id}_right_tactile'][frame_idx]

            for name, img in [('rgb', rgb), ('left_tactile', lt), ('right_tactile', rt)]:
                _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                rr.log(f"camera{cam_id}/{name}",
                       rr.EncodedImage(contents=bytes(buf), media_type="image/jpeg"))

        # --- robot state ---
        for robot_id in range(2):
            pos     = rb.data[f'robot{robot_id}_eef_pos'][frame_idx]               # (3,)
            rot_aa  = rb.data[f'robot{robot_id}_eef_rot_axis_angle'][frame_idx]    # (3,)
            gripper = float(rb.data[f'robot{robot_id}_gripper_width'][frame_idx, 0])

            rot_mat, _ = cv2.Rodrigues(rot_aa)
            rr.log(f"robot{robot_id}/eef",
                   rr.Transform3D(translation=pos.tolist(),
                                  mat3x3=rot_mat.tolist()))
            rr.log(f"robot{robot_id}/eef_pos",
                   rr.Points3D([pos.tolist()], radii=[0.02]))
            rr.log(f"robot{robot_id}/gripper_width",
                   rr.Scalars(gripper))

        global_t += 1

print(f"Saving {OUTPUT_RRD} ...")
rr.save(OUTPUT_RRD)
print("Done.")
