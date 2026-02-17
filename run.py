import time
import threading
from tests.simulators.imu_simulator import ImuSimulator
from tests.simulators.emg_simulator import EmgSimulator
from tests.simulators.mocap_simulator import MocapSimulator
from sensors.imu_outlet import ImuOutlet
from sensors.emg_outlet import EmgOutlet
from sensors.mocap_outlet import MocapOutlet
from sync_engine.sync_engine import SyncEngine, StreamConfig

# 1. Start simulated sensor outlets
imu_out = ImuOutlet(ImuSimulator())
emg_out = EmgOutlet(EmgSimulator())
mocap_out = MocapOutlet(MocapSimulator())

for outlet in [imu_out, emg_out, mocap_out]:
    outlet.start()
    threading.Thread(target=outlet.run, daemon=True).start()

# 2. Start sync engine (discovers outlets automatically via LSL)
engine = SyncEngine([
    StreamConfig("IMU-Knee", "IMU",   num_channels=7,  sample_rate=200),
    StreamConfig("EMG-Quad", "EMG",   num_channels=8,  sample_rate=2000),
    StreamConfig("Mocap",    "Mocap", num_channels=69, sample_rate=120),
])
engine.start()

# 3. Read frames at 60 Hz
while True:
    frame = engine.get_frame()
    if frame:
        print(frame)
    time.sleep(1/60)