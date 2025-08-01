from pylablib.devices import Thorlabs

try:
    # On Linux, you might need to manually specify the device path
    stage = Thorlabs.KinesisMotor("/dev/ttyUSB0") # Replace with the correct device path
    # ... use the stage object for control ...
    stage.move_to(10)
    stage.close()
except Exception as e:
    print(f"Error: {e}")