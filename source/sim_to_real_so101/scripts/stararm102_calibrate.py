"""Calibration entrypoint that registers the project-local Star-Arm adapter."""

import sim_to_real_so101.adapters.stararm102  # noqa: F401
from lerobot.scripts.lerobot_calibrate import main as lerobot_calibrate_main


def main():
    lerobot_calibrate_main()


if __name__ == "__main__":
    main()

