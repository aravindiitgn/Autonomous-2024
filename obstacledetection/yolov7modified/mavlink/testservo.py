from pymavlink import mavutil

# Replace 'com4' with the COM port of your telemetry radio
# Replace 57600 with the correct baud rate if needed
master = mavutil.mavlink_connection('com3')

# Define the system and component IDs (usually 1 for autopilot)
system_id = 1
component_id = 1

# Define the servo channel number for servo_9
SERVO_CHANNEL = 9

# Define the desired angle (in degrees)
DESIRED_ANGLE = 0  # Change this to your desired angle
master.wait_heartbeat()

print("Heartbeat from system (system %u component %u)" % (master.target_system, master.target_component))
# Convert the angle to a servo position value (usually in microseconds)
SERVO_VALUE = int(DESIRED_ANGLE / 180.0 * 1000) + 1000
while 1:
# Send the MAV_CMD_DO_SET_SERVO command to set servo_9
    command = master.mav.command_long_encode(
        system_id,           # System ID
        component_id,        # Component ID
        mavutil.mavlink.MAV_CMD_DO_SET_RELAY,  # Command ID for servo control
        0,                   # Confirmation
        0,       # relay channel number
        1,         # relayon
        0, 0, 0, 0, 0         # Parameters 4-9 (unused)
    )

    master.mav.send(command)
   
    print(command)
    print(DESIRED_ANGLE)

# Close the connection
master.close()
