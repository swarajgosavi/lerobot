from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.utils import busy_wait

camera = OpenCVCamera(camera_index=0, fps=30, width=640, height=480)
camera.connect()
color_image = camera.read()

print(color_image.shape)
print(color_image.dtype)

from lerobot.common.robot_devices.motors.waveshare import WaveshareMotorsBus
follower_port = "/dev/ttyACM0"

follower_arm = WaveshareMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
      "shoulder_pan": (1, "st3215"),
      "shoulder_lift": (2, "st3215"),
      "elbow_flex": (3, "st3215"),
      "wrist_flex": (4, "st3215"),
      "wrist_roll": (5, "sc09_servo"),
      "gripper": (6, "sc09_servo"),
    },
)

# follower_arm.connect()

from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

robot = ManipulatorRobot(
    robot_type="so100",
    # leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/kikobot",
    cameras={
        "laptop": OpenCVCamera(2, fps=30, width=640, height=480),
        # "phone": OpenCVCamera(1, fps=30, width=640, height=480),
    },
)
# robot.disconnect()
robot.connect()

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
import time
import torch

inference_time_s = 60
fps = 30
device = "cpu"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "lerobot/diffusion_pusht"
policy = DiffusionPolicy.from_pretrained(ckpt_path)
policy.to(device)

for _ in range(1 * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)