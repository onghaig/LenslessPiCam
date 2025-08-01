import serial, time, binascii

baud_rates = [921600, 460800, 256000, 230400, 128000,
              115200, 57600, 38400, 19200, 9600]
cmd = bytes.fromhex("05 00 00 00 00 00 05 00")      # “Request LED State”

for br in baud_rates:
    try:
        with serial.Serial("/dev/ttyUSB0", br, timeout=0.25,
                           rtscts=False, dsrdtr=False) as ser:
            ser.reset_input_buffer()
            ser.write(cmd)
            time.sleep(0.05)
            ans = ser.read(16)                       # read up to one whole packet
            print(f"{br:7d} Bd →", binascii.hexlify(ans))
    except Exception as e:
        print(f"{br:7d} Bd → ERROR:", e)
