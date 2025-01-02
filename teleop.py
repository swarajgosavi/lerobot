from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.utils import busy_wait
from pathlib import Path
from huggingface_hub import snapshot_download

from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
follower_port = "/dev/ttyACM0"
leader_port = "/dev/ttyACM1"

leader_arm = FeetechMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
      "shoulder_pan": (1, "sts3215"),
      "shoulder_lift": (2, "sts3215"),
      "elbow_flex": (3, "sts3215"),
      "wrist_flex": (4, "sts3215"),
      "wrist_roll": (5, "sts3215"),
      "gripper": (6, "sts3215"),
    },
)

follower_arm = FeetechMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
      "shoulder_pan": (1, "sts3215"),
      "shoulder_lift": (2, "sts3215"),
      "elbow_flex": (3, "sts3215"),
      "wrist_flex": (4, "sts3215"),
      "wrist_roll": (5, "sts3215"),
      "gripper": (6, "sts3215"),
    },
)

# follower_arm.connect()

from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

robot = ManipulatorRobot(
    robot_type="so100",
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/so100",
    cameras={
        "laptop": OpenCVCamera(0, fps=30, width=640, height=480),
        # "logitech": OpenCVCamera(4, fps=30, width=640, height=480),
    },
)
# robot.disconnect()
robot.connect()

import tqdm
seconds = 120
frequency = 200
for _ in tqdm.tqdm(range(seconds*frequency)):
    leader_pos = robot.leader_arms["main"].read("Present_Position")
    robot.follower_arms["main"].write("Goal_Position", leader_pos)

robot.disconnect()