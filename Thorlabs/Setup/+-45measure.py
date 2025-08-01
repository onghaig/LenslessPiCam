from pylablib.devices import Thorlabs

# use whichever path or serial number just worked for you
stage = Thorlabs.KinesisMotor("/dev/ttyUSB0")          # add is_rack_system=True if you needed it
info = stage.get_device_info()

stage.home()
stage.wait_move()


stage.move_to(-45, channel=None, scale=True)
while stage.get_position(channel=None, scale=True) < 45 :
    
stage.wait_move()

stage.move_by(-5000)    # back to where you started
stage.wait_move()

stage = Thorlabs.KinesisMotor("/dev/ttyUSB0", scale="stage")   # auto‑detects common stages

stage.close()

