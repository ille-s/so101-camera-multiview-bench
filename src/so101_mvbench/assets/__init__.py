"""Asset registry: USDs, textures, robot ArticulationCfg.

Usage:
    from so101_mvbench.assets import ASSETS_DIR
    usd_path = ASSETS_DIR / "usd" / "cylindrical_room_shell.usda"

    from so101_mvbench.assets.so101_white import SO101_WHITE_CFG
"""

from pathlib import Path

ASSETS_DIR: Path = Path(__file__).resolve().parent
