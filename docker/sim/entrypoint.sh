#!/bin/bash
set -e

ISAAC_SIM=/workspace/isaaclab/_isaac_sim

export CARB_APP_PATH=$ISAAC_SIM/kit
export ISAAC_PATH=$ISAAC_SIM
export EXP_PATH=$ISAAC_SIM/apps
source ${ISAAC_SIM}/setup_python_env.sh
source /root/env 2>/dev/null

cat > /usr/local/bin/python << 'WRAPPER'
#!/bin/bash
exec /workspace/isaaclab/_isaac_sim/python.sh "$@"
WRAPPER
chmod +x /usr/local/bin/python

python -m pip install -e /workspace/Sim-to-Real-SO-101-Workshop/source/sim_to_real_so101/
if [ -d /workspace/Sim-to-Real-SO-101-Workshop/rebot/Star-Arm-102/Lerobot/lerobot-teleoperator-stararm102 ]; then
  python -m pip install --no-deps -e /workspace/Sim-to-Real-SO-101-Workshop/rebot/Star-Arm-102/Lerobot/lerobot-teleoperator-stararm102
fi
if [ -d /workspace/Sim-to-Real-SO-101-Workshop/rebot/lerobot-robot-seeed-b601 ]; then
  python -m pip install --no-deps -e /workspace/Sim-to-Real-SO-101-Workshop/rebot/lerobot-robot-seeed-b601
fi

exec "$@"
