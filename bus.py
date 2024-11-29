
from lerobot.common.robot_devices.motors.waveshare import WaveshareMotorsBus

motor_name = "gripper"
motor_index = 6
motor_model = "sc09_servo"

motors_bus = WaveshareMotorsBus (
    port="/dev/ttyACM0",
    motors={motor_name: (motor_index, motor_model)},
)
motors_bus.connect()

position = motors_bus.read("Present_Position")
print(position)

# move from a few motor steps as an example
few_steps = -30
motors_bus.write("Goal_Position", position + few_steps)
print(position + few_steps)

position = motors_bus.read_with_motor_ids(motors_bus.motor_models, motor_index, "ID", num_retry=1)
print(position)

position = motors_bus.read_with_motor_ids(motors_bus.motor_models, motor_index, "Lock", num_retry=1)
print(position)

position = motors_bus.read_with_motor_ids(motors_bus.motor_models, motor_index, "Baud_Rate", num_retry=1)
print(position)

motors_bus.write_with_motor_ids(motors_bus.motor_models, motor_index, "ID", 6, num_retry=1)

position = motors_bus.read_with_motor_ids(motors_bus.motor_models, 6, "ID", num_retry=10)
print(position)

# when done, consider disconnecting
motors_bus.disconnect()