from pylablib.devices import Thorlabs

# use whichever path or serial number just worked for you
stage = Thorlabs.KinesisMotor("/dev/ttyUSB0")          # add is_rack_system=True if you needed it

info = stage.get_device_info()
print("Connected to:", info.model_no, "FW", info.fw_ver, "S/N", info.serial_no)

stage.home()
stage.wait_move()

stage.move_by(5000)     # 5 000 steps forward
stage.wait_move()

stage.move_by(-5000)    # back to where you started
stage.wait_move()


stage = Thorlabs.KinesisMotor("/dev/ttyUSB0", scale="stage")   # auto‑detects common stages
print("Units now:", stage.get_scale_units())

stage.close()
