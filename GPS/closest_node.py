import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import serial
import time
import threading
import re

# Serial port configuration for Arduino
SERIAL_PORT = '/dev/ttyACM1'  # Adjust this to your serial port (e.g., COM3 for Windows, /dev/ttyUSB0 for Linux)
BAUD_RATE = 9600

# Connect to the serial port
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Function to parse NMEA sentences (GPGGA for simplicity)
def parse_gpgga(sentence):
    try:
        if sentence.startswith("$GPGGA"):
            parts = sentence.split(',')
            if len(parts) > 5:
                lat_deg = float(parts[2][:2])
                lat_min = float(parts[2][2:])
                latitude = lat_deg + (lat_min / 60.0)
                if parts[3] == 'S':
                    latitude = -latitude

                lon_deg = float(parts[4][:3])
                lon_min = float(parts[4][3:])
                longitude = lon_deg + (lon_min / 60.0)
                if parts[5] == 'W':
                    longitude = -longitude
                
                return latitude, longitude
    except Exception as e:
        print(f"Error parsing GPGGA: {e}")
    return None, None

# Load your map graph from the OSM file
osm_file_path = "/home/sghost/Downloads/map(8).osm"
graph = ox.graph_from_xml(osm_file_path)

# Function to update GPS position in real-time
# Function to update GPS position in real-time
# Function to update GPS position in real-time
def update_gps_position():
    while True:
        try:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "$GPGGA" in line:
                lat, lon = parse_gpgga(line)
                if lat is not None and lon is not None:
                    print(f"Received Latitude: {lat}, Longitude: {lon}")  # Print received GPS coordinates
                    update_map(lat, lon)
        except Exception as e:
            print(f"Error reading from serial: {e}")

# Function to update the plot with the current GPS position
def update_map(latitude, longitude):
    nearest_node = ox.distance.nearest_nodes(graph, longitude, latitude)
    nearest_node_data = graph.nodes[nearest_node]

    ax.clear()
    ox.plot_graph(graph, ax=ax, show=False, close=False)

    # Add GPS position marker
    ax.plot(longitude, latitude, marker='o', color='red', markersize=10, label="GPS Position")

    # Add nearest node marker
    nearest_node_lon = nearest_node_data['x']
    nearest_node_lat = nearest_node_data['y']
    ax.plot(nearest_node_lon, nearest_node_lat, marker='x', color='blue', markersize=10, label="Nearest Node")

    # Display legend
    ax.legend()

    # Refresh the canvas
    canvas.draw()

# Create a Tkinter window
root = tk.Tk()
root.title("Interactive Map with GPS Position")

# Create a Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ox.plot_graph(graph, ax=ax, show=False, close=False)

# Embed the Matplotlib figure in Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Add Matplotlib's navigation toolbar for zoom and pan functionality
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Start a thread for updating GPS position
thread = threading.Thread(target=update_gps_position, daemon=True)
thread.start()

# Run the Tkinter main loop
root.mainloop()
