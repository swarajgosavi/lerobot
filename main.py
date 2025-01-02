from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

# leader_port = "/dev/ttyACM0"
follower_port = "/dev/ttyACM0"

# leader_arm = DynamixelMotorsBus(
#     port=leader_port,
#     motors={
#         # name: (index, model)
#         "shoulder_pan": (1, "xl330-m077"),
#         "shoulder_lift": (2, "xl330-m077"),
#         "elbow_flex": (3, "xl330-m077"),
#         "wrist_flex": (4, "xl330-m077"),
#         "wrist_roll": (5, "xl330-m077"),
#         "gripper": (6, "xl330-m077"),
#     },
# )

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

follower_arm.connect()

# leader_pos = leader_arm.read("Present_Position")
# follower_pos = follower_arm.read("Present_Position")
# print(leader_pos)
# print(follower_pos)

from lerobot.common.robot_devices.motors.feetech import TorqueMode
import time

follower_arm.write("Torque_Enable", TorqueMode.ENABLED.value)

# Get the current position
position = follower_arm.read("Present_Position")

print(position)

# Update first motor (shoulder_pan) position by +10 steps
# position[0] += 500
# position = [2054] * 6


# Update all motors position by -30 steps
# position -= 30
# follower_arm.write("Goal_Position", position)

# Update gripper by +30 steps
# position[-1] = 0
# follower_arm.write("Goal_Position", position[-1], "shoulder_pan")

time.sleep(2)

# Update gripper by +30 steps
# position[-1] = 50
# follower_arm.write("Goal_Position", position[-1], "shoulder_pan")

follower_arm.write("Torque_Enable", TorqueMode.DISABLED.value)
follower_arm.disconnect()