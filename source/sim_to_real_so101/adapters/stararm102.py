import json
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

from lerobot.motors import MotorCalibration
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS
from lerobot.utils.utils import enter_pressed, move_cursor_up
from lerobot_teleoperator_stararm102.config_stararm102_leader import (
    Stararm102LeaderConfig,
)
from lerobot_teleoperator_stararm102.stararm102_leader import Stararm102Leader


@TeleoperatorConfig.register_subclass("sim_to_real_stararm102")
@dataclass
class SimToRealStararm102LeaderConfig(Stararm102LeaderConfig):
    """Project-local Star-Arm config registered separately from vendor code."""


class SimToRealStararm102Leader(Stararm102Leader):
    """Star-Arm leader with project-specific calibration hardening."""

    config_class = SimToRealStararm102LeaderConfig
    name = "stararm102_leader"

    def __init__(self, config: SimToRealStararm102LeaderConfig):
        self._normalize_calibration_file(config)
        super().__init__(config)

    @classmethod
    def _normalize_calibration_file(cls, config: Stararm102LeaderConfig) -> None:
        calibration_dir = (
            Path(config.calibration_dir)
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / cls.name
        )
        calibration_path = calibration_dir / f"{config.id}.json"
        if not calibration_path.is_file():
            return

        with calibration_path.open() as f:
            calibration = json.load(f)

        changed = False
        for motor_calibration in calibration.values():
            for field in ("id", "drive_mode", "homing_offset", "range_min", "range_max"):
                value = motor_calibration.get(field)
                if isinstance(value, float):
                    motor_calibration[field] = int(round(value))
                    changed = True

        if changed:
            with calibration_path.open("w") as f:
                json.dump(calibration, f, indent=4)
                f.write("\n")

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        self.bus.disable_torque(mode="unlocked")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self._record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=int(round(range_mins[motor])),
                range_max=int(round(range_maxes[motor])),
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def _record_ranges_of_motion(self) -> tuple[dict[str, float], dict[str, float]]:
        motors = list(self.bus.motors)
        positions = {}
        while not positions:
            start_positions = self.bus.sync_read("Present_Position", motors, normalize=False)
            positions = {
                motor: value for motor, value in start_positions.items() if value is not None
            }
            if not positions:
                print("[WARNING]: No valid Star-Arm positions read yet. Check USB/power and keep waiting...")
                time.sleep(0.2)

        mins = positions.copy()
        maxes = positions.copy()
        latest = {motor: positions.get(motor) for motor in motors}

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.bus.sync_read("Present_Position", motors, normalize=False)
            for motor, position in positions.items():
                latest[motor] = position
                if position is None:
                    continue
                mins[motor] = position if motor not in mins else min(position, mins[motor])
                maxes[motor] = position if motor not in maxes else max(position, maxes[motor])

            print("\n-------------------------------------------")
            print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
            for motor in motors:
                min_value = mins.get(motor, "None")
                max_value = maxes.get(motor, "None")
                pos_value = latest.get(motor)
                pos_value = "None" if pos_value is None else pos_value
                print(f"{motor:<15} | {min_value:>6} | {pos_value:>6} | {max_value:>6}")

            if enter_pressed():
                user_pressed_enter = True

            if not user_pressed_enter:
                move_cursor_up(len(motors) + 3)

        missing = [motor for motor in motors if motor not in mins or motor not in maxes]
        if missing:
            raise ValueError(
                "Some motors never returned a valid position during calibration:\n"
                f"{pformat(missing)}"
            )

        same_min_max = [motor for motor in motors if mins[motor] == maxes[motor]]
        if same_min_max:
            raise ValueError(f"Some motors have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

