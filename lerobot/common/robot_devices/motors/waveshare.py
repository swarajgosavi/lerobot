import enum
import logging
import math
import time
import traceback
from copy import deepcopy

import numpy as np
import tqdm

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

PROTOCOL_VERSION_0 = 0
PROTOCOL_VERSION_1 = 1
BAUDRATE = 1_000_000
TIMEOUT_MS = 1000

MAX_ID_RANGE = 252

# The following bounds define the lower and upper joints range (after calibration).
# For joints in degree (i.e. revolute joints), their nominal range is [-180, 180] degrees
# which corresponds to a half rotation on the left and half rotation on the right.
# Some joints might require higher range, so we allow up to [-270, 270] degrees until
# an error is raised.
LOWER_BOUND_DEGREE = -270
UPPER_BOUND_DEGREE = 270
# For joints in percentage (i.e. joints that move linearly like the prismatic joint of a gripper),
# their nominal range is [0, 100] %. For instance, for Aloha gripper, 0% is fully
# closed, and 100% is fully open. To account for slight calibration issue, we allow up to
# [-10, 110] until an error is raised.
LOWER_BOUND_LINEAR = -10
UPPER_BOUND_LINEAR = 110

HALF_TURN_DEGREE = 180

# See this link for ST3215 Memory Table:
# https://files.waveshare.com/upload/2/27/ST3215%20memory%20register%20map-EN.xls
# data_name: (address, size_byte)
ST3215_SERIES_CONTROL_TABLE = {
    "Model": (3, 2),
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Angle_Limit": (9, 2),
    "Max_Angle_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1),
    "Offset": (31, 2),
    "Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Speed_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Speed": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Speed": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    # Not in the Memory Table
    "Maximum_Acceleration": (85, 2),
}

SCS_SERIES_BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
    3: 128_000,
    4: 115_200,
    5: 57_600,
    6: 38_400,
    7: 19_200,
}

CALIBRATION_REQUIRED = ["Goal_Position", "Present_Position"]
CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]


MODEL_CONTROL_TABLE = {
    "st3215" : ST3215_SERIES_CONTROL_TABLE,
    "sc09_servo": ST3215_SERIES_CONTROL_TABLE,
}

MODEL_RESOLUTION = {
    "st3215": 4096,
    "sc09_servo": 1024,
}

MODEL_BAUDRATE_TABLE = {
    "st3215": SCS_SERIES_BAUDRATE_TABLE,
    "sc09_servo": SCS_SERIES_BAUDRATE_TABLE,
}

# High number of retries is needed for feetech compared to dynamixel motors.
NUM_READ_RETRY = 20
NUM_WRITE_RETRY = 20

# def convert_degrees_to_steps(degrees: float | np.ndarray, models: str | list[str]) -> np.ndarray:
#     """This function converts the degree range to the step range for indicating motors rotation.
#     It assumes a motor achieves a full rotation by going from -180 degree position to +180.
#     The motor resolution (e.g. 4096) corresponds to the number of steps needed to achieve a full rotation.
#     """
#     resolutions = [MODEL_RESOLUTION[model] for model in models]
#     steps = degrees / 180 * np.array(resolutions) / 2
#     steps = steps.astype(int)
#     return steps

# def convert_degrees_to_steps_sc09(degrees: float | np.ndarray, models: str | list[str]) -> np.ndarray:
#     """This function converts the degree range to the step range for indicating motors rotation.
#     It assumes a motor achieves a full rotation by going from -150 degree position to +150.
#     The motor resolution (e.g. 1024) corresponds to the number of steps needed to achieve a partial rotation.
#     """
#     resolutions = [MODEL_RESOLUTION[model] for model in models]
#     steps = degrees / 150 * np.array(resolutions) / 2
#     steps = steps.astype(int)
#     return steps

def convert_degrees_to_steps(degrees: float | np.ndarray, models: str | list[str]) -> np.ndarray:
    """
    Converts the degree range to the step range for indicating motor rotation.
    Assumes different motors can have varying resolutions and rotation angles.

    Args:
        degrees (float | np.ndarray): The angle(s) in degrees to be converted to steps.
        models (str | list[str]): The motor model(s) (e.g., "sc09_servo").

    Returns:
        np.ndarray: Corresponding step values for the given degree angles.
    """
    # Ensure `models` is a list for consistency
    if isinstance(models, str):
        models = [models]

    steps = []
    for model in models:
        resolution = MODEL_RESOLUTION[model]  # Get resolution for the motor
        if model == "sc09_servo":
            # SC09 has a 300° range mapped to 1024 steps
            steps.append(degrees / 300 * resolution)
        else:
            # Default: 360° range mapped to the model resolution
            steps.append(degrees / 360 * resolution)

    # Convert to integer steps and return
    return np.round(steps).astype(int)

def convert_to_bytes(value, bytes, mock=False):
    if mock:
        return value

    import scservo_sdk as scs

    # Note: No need to convert back into unsigned int, since this byte preprocessing
    # already handles it for us.
    if bytes == 1:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
        ]
    elif bytes == 2:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
        ]
    elif bytes == 4:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
        ]
    else:
        raise NotImplementedError(
            f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but "
            f"{bytes} is provided instead."
        )
    return data


def get_group_sync_key(data_name, motor_names):
    group_key = f"{data_name}_" + "_".join(motor_names)
    return group_key


def get_result_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    rslt_name = f"{fn_name}_{group_key}"
    return rslt_name


def get_queue_name(fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    queue_name = f"{fn_name}_{group_key}"
    return queue_name


def get_log_name(var_name, fn_name, data_name, motor_names):
    group_key = get_group_sync_key(data_name, motor_names)
    log_name = f"{var_name}_{fn_name}_{group_key}"
    return log_name


def assert_same_address(model_ctrl_table, motor_models, data_name):
    all_addr = []
    all_bytes = []
    for model in motor_models:
        addr, bytes = model_ctrl_table[model][data_name]
        all_addr.append(addr)
        all_bytes.append(bytes)

    if len(set(all_addr)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different address for `data_name`='{data_name}' ({list(zip(motor_models, all_addr, strict=False))}). Contact a LeRobot maintainer."
        )

    if len(set(all_bytes)) != 1:
        raise NotImplementedError(
            f"At least two motor models use a different bytes representation for `data_name`='{data_name}' ({list(zip(motor_models, all_bytes, strict=False))}). Contact a LeRobot maintainer."
        )


class TorqueMode(enum.Enum):
    ENABLED = 1
    DISABLED = 0


class DriveMode(enum.Enum):
    NON_INVERTED = 0
    INVERTED = 1


class CalibrationMode(enum.Enum):
    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    DEGREE = 0
    # Joints with linear motions (like gripper of Aloha) are experessed in nominal range of [0, 100]
    LINEAR = 1


class JointOutOfRangeError(Exception):
    def __init__(self, message="Joint is out of range"):
        self.message = message
        super().__init__(self.message)

class WaveshareMotorsBus:

    def __init__(
        self,
        port: str,
        motors: dict[str, tuple[int, str]],
        extra_model_control_table: dict[str, list[tuple]] | None = None,
        extra_model_resolution: dict[str, int] | None = None,
        mock=False,
    ):
        self.port = port
        self.motors = motors
        self.mock = mock

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        if extra_model_control_table:
            self.model_ctrl_table.update(extra_model_control_table)

        self.model_resolution = deepcopy(MODEL_RESOLUTION)
        if extra_model_resolution:
            self.model_resolution.update(extra_model_resolution)

        self.port_handler = None
        self.packet_handler = None

        self.packet_handler_0 = None
        self.packet_handler_1 = None

        self.calibration = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}

        self.track_positions = {}

        print("init", motors)

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"FeetechMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )

        if self.mock:
            import tests.mock_scservo_sdk as scs
        else:
            import scservo_sdk as scs

        self.port_handler = scs.PortHandler(self.port)
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_0)
        # self.packet_handler_0 = scs.PacketHandler(PROTOCOL_VERSION_0)
        # self.packet_handler_1 = scs.PacketHandler(PROTOCOL_VERSION_1)

        try:
            if not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
        except Exception:
            traceback.print_exc()
            print(
                "\nTry running `python lerobot/scripts/find_motors_bus_port.py` to make sure you are using the correct port.\n"
            )
            raise

        # Allow to read and write
        self.is_connected = True

        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)

    def reconnect(self):
        if self.mock:
            import tests.mock_scservo_sdk as scs
        else:
            import scservo_sdk as scs

        self.port_handler = scs.PortHandler(self.port)
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_0)
        # self.packet_handler_0 = scs.PacketHandler(PROTOCOL_VERSION_0)
        # self.packet_handler_1 = scs.PacketHandler(PROTOCOL_VERSION_1)

        if not self.port_handler.openPort():
            raise OSError(f"Failed to open port '{self.port}'.")

        self.is_connected = True

    def are_motors_configured(self):
        # Only check the motor indices and not baudrate, since if the motor baudrates are incorrect,
        # a ConnectionError will be raised anyway.
        try:
            return (self.motor_indices == self.read("ID")).all()
        except ConnectionError as e:
            print(e)
            return False

    def find_motor_indices(self, possible_ids=None, num_retry=2):
        if possible_ids is None:
            possible_ids = range(MAX_ID_RANGE)

        import scservo_sdk as scs

        indices = []
        print("Searching for servo ID...")
        for test_id in tqdm.tqdm(possible_ids):  # IDs range from 0 to 253
            try:
                # Ping the servo
                model_number, comm_result, error = self.packet_handler.ping(self.port_handler, test_id)
                if comm_result == scs.COMM_SUCCESS:
                    print(f"Servo found! ID: {test_id}, Model Number: {model_number}")
                    indices.append(test_id)
                elif error:
                    print(f"Error from servo ID {test_id}: {self.packet_handler.getRxPacketError(error)}")
            except Exception as e:
                print(f"An error occurred while pinging ID {test_id}: {e}")
        # for idx in tqdm.tqdm(possible_ids):
        #     try:
        #         present_idx = self.read_with_motor_ids(self.motor_models, [idx], "ID", num_retry=num_retry)[0]
        #     except ConnectionError:
        #         continue

        #     if idx != present_idx:
        #         # sanity check
        #         raise OSError(
        #             "Motor index used to communicate through the bus is not the same as the one present in the motor memory. The motor memory might be damaged."
        #         )
        #     indices.append(idx)

        return indices

    def set_bus_baudrate(self, baudrate):
        present_bus_baudrate = self.port_handler.getBaudRate()
        if present_bus_baudrate != baudrate:
            print(f"Setting bus baud rate to {baudrate}. Previously {present_bus_baudrate}.")
            self.port_handler.setBaudRate(baudrate)

            if self.port_handler.getBaudRate() != baudrate:
                raise OSError("Failed to write bus baud rate.")
            
    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict[str, list]):
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values: np.ndarray | list, motor_names: list[str] | None):
        """This function apply the calibration, automatically detects out of range errors for motors values and attempt to correct.

        For more info, see docstring of `apply_calibration` and `autocorrect_calibration`.
        """
        try:
            values = self.apply_calibration(values, motor_names)
        except JointOutOfRangeError as e:
            print(e)
            self.autocorrect_calibration(values, motor_names)
            values = self.apply_calibration(values, motor_names)
        return values
    
    def apply_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 nominal degree range ]-180.0, 180.0[ with
        a "zero position" at 0 degree.

        Note: We say "nominal degree range" since the motors can take values outside this range. For instance, 190 degrees, if the motor
        rotate more than a half a turn from the zero position. However, most motors can't rotate more than 180 degrees and will stay in this range.

        Joints values are original in [0, 2**32[ (unsigned int32). Each motor are expected to complete a full rotation
        when given a goal position that is + or - their resolution. For instance, feetech xl330-m077 have a resolution of 4096, and
        at any position in their original range, let's say the position 56734, they complete a full rotation clockwise by moving to 60830,
        or anticlockwise by moving to 52638. The position in the original range is arbitrary and might change a lot between each motor.
        To harmonize between motors of the same model, different robots, or even models of different brands, we propose to work
        in the centered nominal degree range ]-180, 180[.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Update direction of rotation of the motor to match between leader and follower.
                # In fact, the motor of the leader for a given joint can be assembled in an
                # opposite direction in term of rotation than the motor of the follower on the same joint.
                if drive_mode:
                    values[i] *= -1

                # Convert from range [-2**31, 2**31[ to
                # nominal range ]-resolution, resolution[ (e.g. ]-2048, 2048[)
                values[i] += homing_offset

                # Convert from range ]-resolution, resolution[ to
                # universal float32 centered degree range ]-180, 180[
                values[i] = values[i] / (resolution // 2) * HALF_TURN_DEGREE

                if (values[i] < LOWER_BOUND_DEGREE) or (values[i] > UPPER_BOUND_DEGREE):
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for {name}. "
                        f"Expected to be in nominal range of [-{HALF_TURN_DEGREE}, {HALF_TURN_DEGREE}] degrees (a full rotation), "
                        f"with a maximum range of [{LOWER_BOUND_DEGREE}, {UPPER_BOUND_DEGREE}] degrees to account for joints that can rotate a bit more, "
                        f"but present value is {values[i]} degree. "
                        "This might be due to a cable connection issue creating an artificial 360 degrees jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the present position to a nominal range [0, 100] %,
                # useful for joints with linear motions like Aloha gripper
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    raise JointOutOfRangeError(
                        f"Wrong motor position range detected for {name}. "
                        f"Expected to be in nominal range of [0, 100] % (a full linear translation), "
                        f"with a maximum range of [{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}] % to account for some imprecision during calibration, "
                        f"but present value is {values[i]} %. "
                        "This might be due to a cable connection issue creating an artificial jump in motor values. "
                        "You need to recalibrate by running: `python lerobot/scripts/control_robot.py calibrate`"
                    )

        return values

    def autocorrect_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """This function automatically detects issues with values of motors after calibration, and correct for these issues.

        Some motors might have values outside of expected maximum bounds after calibration.
        For instance, for a joint in degree, its value can be outside [-270, 270] degrees, which is totally unexpected given
        a nominal range of [-180, 180] degrees, which represents half a turn to the left or right starting from zero position.

        Known issues:
        #1: Motor value randomly shifts of a full turn, caused by hardware/connection errors.
        #2: Motor internal homing offset is shifted of a full turn, caused by using default calibration (e.g Aloha).
        #3: motor internal homing offset is shifted of less or more than a full turn, caused by using default calibration
            or by human error during manual calibration.

        Issues #1 and #2 can be solved by shifting the calibration homing offset by a full turn.
        Issue #3 will be visually detected by user and potentially captured by the safety feature `max_relative_target`,
        that will slow down the motor, raise an error asking to recalibrate. Manual recalibrating will solve the issue.

        Note: A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert from unsigned int32 original range [0, 2**32] to signed float32 range
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                if drive_mode:
                    values[i] *= -1

                # Convert from initial range to range [-180, 180] degrees
                calib_val = (values[i] + homing_offset) / (resolution // 2) * HALF_TURN_DEGREE
                in_range = (calib_val > LOWER_BOUND_DEGREE) and (calib_val < UPPER_BOUND_DEGREE)

                # Solve this inequality to find the factor to shift the range into [-180, 180] degrees
                # values[i] = (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE
                # - HALF_TURN_DEGREE <= (values[i] + homing_offset + resolution * factor) / (resolution // 2) * HALF_TURN_DEGREE <= HALF_TURN_DEGREE
                # (- HALF_TURN_DEGREE / HALF_TURN_DEGREE * (resolution // 2) - values[i] - homing_offset) / resolution <= factor <= (HALF_TURN_DEGREE / 180 * (resolution // 2) - values[i] - homing_offset) / resolution
                low_factor = (
                    -HALF_TURN_DEGREE / HALF_TURN_DEGREE * (resolution // 2) - values[i] - homing_offset
                ) / resolution
                upp_factor = (
                    HALF_TURN_DEGREE / HALF_TURN_DEGREE * (resolution // 2) - values[i] - homing_offset
                ) / resolution

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from initial range to range [0, 100] in %
                calib_val = (values[i] - start_pos) / (end_pos - start_pos) * 100
                in_range = (calib_val > LOWER_BOUND_LINEAR) and (calib_val < UPPER_BOUND_LINEAR)

                # Solve this inequality to find the factor to shift the range into [0, 100] %
                # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos + resolution * factor - start_pos - resolution * factor) * 100
                # values[i] = (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100
                # 0 <= (values[i] - start_pos + resolution * factor) / (end_pos - start_pos) * 100 <= 100
                # (start_pos - values[i]) / resolution <= factor <= (end_pos - values[i]) / resolution
                low_factor = (start_pos - values[i]) / resolution
                upp_factor = (end_pos - values[i]) / resolution

            if not in_range:
                # Get first integer between the two bounds
                if low_factor < upp_factor:
                    factor = math.ceil(low_factor)

                    if factor > upp_factor:
                        raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")
                else:
                    factor = math.ceil(upp_factor)

                    if factor > low_factor:
                        raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")

                if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                    out_of_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
                    in_range_str = f"{LOWER_BOUND_DEGREE} < {calib_val} < {UPPER_BOUND_DEGREE} degrees"
                elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                    out_of_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"
                    in_range_str = f"{LOWER_BOUND_LINEAR} < {calib_val} < {UPPER_BOUND_LINEAR} %"

                logging.warning(
                    f"Auto-correct calibration of motor '{name}' by shifting value by {abs(factor)} full turns, "
                    f"from '{out_of_range_str}' to '{in_range_str}'."
                )

                # A full turn corresponds to 360 degrees but also to 4096 steps for a motor resolution of 4096.
                self.calibration["homing_offset"][calib_idx] += resolution * factor

    def revert_calibration(self, values: np.ndarray | list, motor_names: list[str] | None):
        """Inverse of `apply_calibration`."""
        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Convert from nominal 0-centered degree range [-180, 180] to
                # 0-centered resolution range (e.g. [-2048, 2048] for resolution=4096)
                values[i] = values[i] / HALF_TURN_DEGREE * (resolution // 2)

                # Substract the homing offsets to come back to actual motor range of values
                # which can be arbitrary.
                values[i] -= homing_offset

                # Remove drive mode, which is the rotation direction of the motor, to come back to
                # actual motor rotation direction which can be arbitrary.
                if drive_mode:
                    values[i] *= -1

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from nominal lnear range of [0, 100] % to
                # actual motor range of values which can be arbitrary.
                values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

        values = np.round(values).astype(np.int32)
        return values

    def avoid_rotation_reset(self, values, motor_names, data_name):
        if data_name not in self.track_positions:
            self.track_positions[data_name] = {
                "prev": [None] * len(self.motor_names),
                # Assume False at initialization
                "below_zero": [False] * len(self.motor_names),
                "above_max": [False] * len(self.motor_names),
            }

        track = self.track_positions[data_name]

        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            idx = self.motor_names.index(name)

            if track["prev"][idx] is None:
                track["prev"][idx] = values[i]
                continue

            # Detect a full rotation occured
            if abs(track["prev"][idx] - values[i]) > 2048:
                # Position went below 0 and got reset to 4095
                if track["prev"][idx] < values[i]:
                    # So we set negative value by adding a full rotation
                    values[i] -= 4096

                # Position went above 4095 and got reset to 0
                elif track["prev"][idx] > values[i]:
                    # So we add a full rotation
                    values[i] += 4096

            track["prev"][idx] = values[i]

        return values
    
    def apply_calibration_1(self, values: np.ndarray | list, motor_names: list[str] | None):
        """
        Convert from unsigned int32 joint position range [0, 2**32[ to the universal float32 nominal degree range.
        For SC09, the range is adjusted to [-150.0, 150.0] degrees to account for its 300° rotation range.

        Args:
            values (np.ndarray | list): The joint positions in the motor's original range.
            motor_names (list[str] | None): List of motor names to calibrate.

        Returns:
            np.ndarray: Calibrated joint positions in degrees.
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert input values to float32 for calculations
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                if model == "sc09_servo":
                    # SC09-specific: Adjust the nominal degree range [-150.0, 150.0]
                    nominal_range = 150.0
                else:
                    # Default: Full rotation [-180.0, 180.0]
                    nominal_range = HALF_TURN_DEGREE

                # Reverse direction if needed
                if drive_mode:
                    values[i] *= -1

                # Apply homing offset
                values[i] += homing_offset

                # Convert steps to degrees within the adjusted range
                values[i] = (values[i] / (resolution / 2)) * nominal_range

                # Validate if the value falls within acceptable degree bounds
                lower_bound = -nominal_range - 30  # Allow slight over-rotation
                upper_bound = nominal_range + 30
                if (values[i] < lower_bound) or (values[i] > upper_bound):
                    raise JointOutOfRangeError(
                        f"Motor position out of range for {name}. "
                        f"Expected [-{nominal_range}, {nominal_range}] degrees "
                        f"(with maximum allowed range [{lower_bound}, {upper_bound}]). "
                        f"Current value: {values[i]} degrees."
                    )

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the position to the nominal range [0, 100] %
                values[i] = (values[i] - start_pos) / (end_pos - start_pos) * 100

                # Validate if the value falls within acceptable bounds
                if (values[i] < LOWER_BOUND_LINEAR) or (values[i] > UPPER_BOUND_LINEAR):
                    raise JointOutOfRangeError(
                        f"Linear position out of range for {name}. "
                        f"Expected [0, 100]% (with maximum allowed range "
                        f"[{LOWER_BOUND_LINEAR}, {UPPER_BOUND_LINEAR}]). "
                        f"Current value: {values[i]}%."
                    )

        return values

    def autocorrect_calibration_1(self, values: np.ndarray | list, motor_names: list[str] | None):
        """
        Automatically detects and corrects calibration issues for SC09 or other motors.

        For SC09:
        - Adjusts the nominal range to [-150.0, 150.0] degrees.
        - Handles issues with out-of-bound positions by shifting calibration offsets.

        Args:
            values (np.ndarray | list): Current motor positions in steps.
            motor_names (list[str] | None): List of motor names to autocorrect.

        Returns:
            None
        """
        if motor_names is None:
            motor_names = self.motor_names

        # Convert to float32 for calculations
        values = values.astype(np.float32)

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Define nominal range based on model
                if model == "sc09_servo":
                    nominal_range = 150.0  # SC09 range in degrees
                else:
                    nominal_range = HALF_TURN_DEGREE  # Default: 180.0 degrees

                # Reverse direction if needed
                if drive_mode:
                    values[i] *= -1

                # Convert to degrees in the nominal range
                calib_val = (values[i] + homing_offset) / (resolution / 2) * nominal_range
                in_range = (-nominal_range <= calib_val <= nominal_range)

                # Solve to find the factor to shift into range
                low_factor = (-nominal_range - calib_val) / (resolution / 2)
                upp_factor = (nominal_range - calib_val) / (resolution / 2)

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Rescale the position to a percentage range
                calib_val = (values[i] - start_pos) / (end_pos - start_pos) * 100
                in_range = (0 <= calib_val <= 100)

                # Solve to find the factor to shift into range
                low_factor = (0 - calib_val) / resolution
                upp_factor = (100 - calib_val) / resolution

            if not in_range:
                # Find the smallest integer factor to bring the value into range
                factor = math.ceil(low_factor) if low_factor < upp_factor else math.ceil(upp_factor)

                if low_factor > upp_factor or upp_factor > low_factor:
                    raise ValueError(f"No integer found between bounds [{low_factor=}, {upp_factor=}]")

                # Adjust the homing offset to bring the value into range
                self.calibration["homing_offset"][calib_idx] += resolution * factor

                logging.warning(
                    f"Auto-corrected motor '{name}' by shifting calibration offset by {abs(factor)} full turns. "
                    f"New range: {calib_val} degrees."
                )

    def revert_calibration_1(self, values: np.ndarray | list, motor_names: list[str] | None):
        """
        Inverse of `apply_calibration`. Converts degrees or percentages back to motor steps.

        Args:
            values (np.ndarray | list): Calibrated values (degrees or percentages).
            motor_names (list[str] | None): Names of motors to apply the inverse calibration.

        Returns:
            np.ndarray: Motor step values.
        """
        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            calib_idx = self.calibration["motor_names"].index(name)
            calib_mode = self.calibration["calib_mode"][calib_idx]

            if CalibrationMode[calib_mode] == CalibrationMode.DEGREE:
                drive_mode = self.calibration["drive_mode"][calib_idx]
                homing_offset = self.calibration["homing_offset"][calib_idx]
                _, model = self.motors[name]
                resolution = self.model_resolution[model]

                # Determine nominal range based on model
                if model == "sc09_servo":
                    nominal_range = 150.0  # SC09 range in degrees
                else:
                    nominal_range = HALF_TURN_DEGREE  # Default: 180.0 degrees

                # Convert from degrees to steps in the motor's resolution
                values[i] = values[i] / nominal_range * (resolution / 2)

                # Subtract homing offset to revert to motor's raw step value
                values[i] -= homing_offset

                # Revert drive mode (direction) if enabled
                if drive_mode:
                    values[i] *= -1

            elif CalibrationMode[calib_mode] == CalibrationMode.LINEAR:
                start_pos = self.calibration["start_pos"][calib_idx]
                end_pos = self.calibration["end_pos"][calib_idx]

                # Convert from percentage back to motor step range
                values[i] = values[i] / 100 * (end_pos - start_pos) + start_pos

        values = np.round(values).astype(np.int32)
        return values
    
    def avoid_rotation_reset_1(self, values, motor_names, data_name):
        """
        Adjusts motor step values to prevent rotation reset issues caused by hardware limitations.

        Args:
            values (np.ndarray): Motor step values.
            motor_names (list[str]): Names of the motors to process.
            data_name (str): The data parameter being tracked.

        Returns:
            np.ndarray: Adjusted motor step values.
        """
        if data_name not in self.track_positions:
            self.track_positions[data_name] = {
                "prev": [None] * len(self.motor_names),
                "below_zero": [False] * len(self.motor_names),
                "above_max": [False] * len(self.motor_names),
            }

        track = self.track_positions[data_name]

        if motor_names is None:
            motor_names = self.motor_names

        for i, name in enumerate(motor_names):
            idx = self.motor_names.index(name)
            _, model = self.motors[name]
            resolution = self.model_resolution[model]

            # Initialize tracking if this is the first value
            if track["prev"][idx] is None:
                track["prev"][idx] = values[i]
                continue

            # Detect a full rotation reset specific to SC09
            threshold = resolution // 2  # Half of resolution (e.g., 512 for SC09)
            if abs(track["prev"][idx] - values[i]) > threshold:
                if track["prev"][idx] < values[i]:
                    # Position reset to 0, add full rotation backward
                    values[i] -= resolution
                elif track["prev"][idx] > values[i]:
                    # Position reset to max, add full rotation forward
                    values[i] += resolution

            # Update tracking
            track["prev"][idx] = values[i]

        return values

    def read_with_motor_ids(self, motor_models, motor_ids, data_name, num_retry=NUM_READ_RETRY):
        if self.mock:
            import tests.mock_scservo_sdk as scs
        else:
            import scservo_sdk as scs

        return_list = True
        if not isinstance(motor_ids, list):
            return_list = False
            motor_ids = [motor_ids]

        motor_names = None
        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        models = []
        st = []
        sc = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            models.append(model)
            # if motor_idx in motor_ids:
            if model == "st3215":
                st.append(motor_idx)
            elif model == "sc09_servo":
                sc.append(motor_idx)

        assert_same_address(self.model_ctrl_table, self.motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
        group = scs.GroupSyncRead(self.port_handler, self.packet_handler, addr, bytes)

        values = []
        if len(st) != 0:
            self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_0)
            for idx in st:
                group.addParam(idx)

            for _ in range(num_retry):
                comm = group.txRxPacket()
                if comm == scs.COMM_SUCCESS:
                    break

            if comm != scs.COMM_SUCCESS:
                raise ConnectionError(
                    f"Read failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                    f"{self.packet_handler.getTxRxResult(comm)}"
                )

            # values = []
            for idx in motor_ids:
                value = group.getData(idx, addr, bytes)
                values.append(value)
        
        elif len(sc) != 0:
            self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_1)
            for idx in sc:
                # Read SCServo present position
                scs_present_position_speed, scs_comm_result, scs_error = self.packet_handler.read4ByteTxRx(self.port_handler, idx, addr)
                if scs_comm_result != scs.COMM_SUCCESS:
                    print("result", self.packet_handler.getTxRxResult(scs_comm_result))
                elif scs_error != 0:
                    print("error", self.packet_handler.getRxPacketError(scs_error))

                scs_present_position = scs.SCS_LOWORD(scs_present_position_speed)
                scs_present_speed = scs.SCS_HIWORD(scs_present_position_speed)

                print(scs_present_position)

                value = scs_present_position
                values.append(value)

        if return_list:
            return values
        else:
            return values[0]


    def read(self, data_name, motor_names: str | list[str] | None = None):
        if self.mock:
            import tests.mock_scservo_sdk as scs
        else:
            import scservo_sdk as scs

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FeetechMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        motor_ids = []
        models = []
        st = []
        sc = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)
            if model == "st3215":
                st.append(motor_idx)
            elif model == "sc09_servo":
                sc.append(motor_idx)


        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        values = []
        if len(st) != 0:
            self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_0)
            if data_name not in self.group_readers:
                # create new group reader
                self.group_readers[group_key] = scs.GroupSyncRead(
                    self.port_handler, self.packet_handler, addr, bytes
                )
                for idx in st:
                    self.group_readers[group_key].addParam(idx)

            for _ in range(NUM_READ_RETRY):
                comm = self.group_readers[group_key].txRxPacket()
                if comm == scs.COMM_SUCCESS:
                    break

            if comm != scs.COMM_SUCCESS:
                raise ConnectionError(
                    f"Read failed due to communication error on port {self.port} for group_key {group_key}: "
                    f"{self.packet_handler.getTxRxResult(comm)}"
                )

            # values = []
            for idx in motor_ids:
                value = self.group_readers[group_key].getData(idx, addr, bytes)
                values.append(value)
        elif len(sc) != 0:
            self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_1)
            for idx in sc:
                # Read SCServo present position
                scs_present_position_speed, scs_comm_result, scs_error = self.packet_handler.read4ByteTxRx(self.port_handler, idx, addr)
                if scs_comm_result != scs.COMM_SUCCESS:
                    print(self.packet_handler.getTxRxResult(scs_comm_result))
                elif scs_error != 0:
                    print(self.packet_handler.getRxPacketError(scs_error))

                scs_present_position = scs.SCS_LOWORD(scs_present_position_speed)
                scs_present_speed = scs.SCS_HIWORD(scs_present_position_speed)

                print(scs_present_position, scs_present_speed)

                value = scs_present_position
                values.append(value)


        values = np.array(values)

        # Convert to signed int to use range [-2048, 2048] for our motor positions.
        if data_name in CONVERT_UINT32_TO_INT32_REQUIRED:
            values = values.astype(np.int32)

        if data_name in CALIBRATION_REQUIRED:
            values = self.avoid_rotation_reset(values, motor_names, data_name)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.apply_calibration_autocorrect(values, motor_names)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def write_with_motor_ids(self, motor_models, motor_ids, data_name, values, num_retry=NUM_WRITE_RETRY):
        if self.mock:
            import tests.mock_scservo_sdk as scs
        else:
            import scservo_sdk as scs

        if not isinstance(motor_ids, list):
            motor_ids = [motor_ids]
        if not isinstance(values, list):
            values = [values]

        motor_names = None
        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        models = []
        st = []
        sc = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            # motor_ids.append(motor_idx)
            models.append(model)
            if model == "st3215":
                st.append(motor_idx)
            elif model == "sc09_servo":
                sc.append(motor_idx)

        assert_same_address(self.model_ctrl_table, motor_models, data_name)
        addr, bytes = self.model_ctrl_table[motor_models[0]][data_name]
#  error prone
        for motor_ids in (st, sc):
            
            if len(motor_ids) == 0:
                continue

            if motor_ids == st:
                self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_0)
            elif motor_ids == sc:
                self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_1)

            print(motor_names)
            group = scs.GroupSyncWrite(self.port_handler, self.packet_handler, addr, bytes)
            for idx, value in zip(motor_ids, values, strict=True):
                data = convert_to_bytes(value, bytes, self.mock)
                group.addParam(idx, data)

            for _ in range(num_retry):
                comm = group.txPacket()
                if comm == scs.COMM_SUCCESS:
                    break

            if comm != scs.COMM_SUCCESS:
                raise ConnectionError(
                    f"Write failed due to communication error on port {self.port_handler.port_name} for indices {motor_ids}: "
                    f"{self.packet_handler.getTxRxResult(comm)}"
                )
            print("Done")


    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FeetechMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        if self.mock:
            import tests.mock_scservo_sdk as scs
        else:
            import scservo_sdk as scs

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float, np.integer)):
            values = [int(values)] * len(motor_names)

        values = np.array(values)

        motor_ids = []
        models = []
        st = []
        sc = []
        for name in motor_names:
            motor_idx, model = self.motors[name]
            motor_ids.append(motor_idx)
            models.append(model)
            if model == "st3215":
                st.append(motor_idx)
            elif model == "sc09_servo":
                sc.append(motor_idx)

        if data_name in CALIBRATION_REQUIRED and self.calibration is not None:
            values = self.revert_calibration(values, motor_names)

        values = values.tolist()

        assert_same_address(self.model_ctrl_table, models, data_name)
        addr, bytes = self.model_ctrl_table[model][data_name]
        group_key = get_group_sync_key(data_name, motor_names)

        for motor_ids in (st, sc):
            
            if len(motor_ids) == 0:
                continue

            if motor_ids == st:
                self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_0)
            elif motor_ids == sc:
                self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION_1)

# error prone
            init_group = data_name not in self.group_readers
            if init_group:
                self.group_writers[group_key] = scs.GroupSyncWrite(
                    self.port_handler, self.packet_handler, addr, bytes
                )

            for idx, value in zip(motor_ids, values, strict=True):
                data = convert_to_bytes(value, bytes, self.mock)
                if init_group:
                    self.group_writers[group_key].addParam(idx, data)
                else:
                    self.group_writers[group_key].changeParam(idx, data)

            comm = self.group_writers[group_key].txPacket()
            if comm != scs.COMM_SUCCESS:
                raise ConnectionError(
                    f"Write failed due to communication error on port {self.port} for group_key {group_key}: "
                    f"{self.packet_handler.getTxRxResult(comm)}"
                )

        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FeetechMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        if self.port_handler is not None:
            self.port_handler.closePort()
            self.port_handler = None

        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()