# Autonomous-2024
This project is a result of a full semester course **ME 691-XIII Autonomous Vehicles** offered at IIT Gandhinagar by prof Harish PM. The project is completely student run and still under progress, there are further developments on the way. We have acheived Level 3 Autonomy on an Electric Golf cart with very less hardware changes.

There are multiple sections of the cart on which modifications were done.
## Perception

### Hardware
### Software

## Steering Control

## Braking

## Localization
### Hardware
Used an Arduino UNO R3 and a Neo 6mv2 GPS Module. The accuracy of this GPS module is close to 2.5~3.5 m while the GPS is fixed to atleast 5 satellites for trilateration.
### Software
So our aim is to run the cart autonomously anywhere inside our campus. For that we need to know where is it??
One of the difficulties was to get a map of the campus roads but thanks to OpenStreetMaps - ( an open source maps library) that made our jobs much easier. OpenStreetMaps comes with **osmnx** - a library that u can install directly via pip to your python and start using their functionalities like projection and map inference. 

First we got the map of interest from their Website( Essentially an xml file that contains maps and its lat,long projections). Simply download the map and next we used a library called **networkx** to convert that map to a Graph of nodes and edges. That helps us in applying any path finding algorithm easily like djikstra or A*.

Next we have a map of the campus roads visible through a tkinter gui window on which we select the Start and end node and a* algorithm returns a path that the cart can traverse. Meanwhile we can also show our current location on the map by projecting the latitude, longitude values coming from the GPS Module back to the map by **pyproj** library. ![Alt text][.

### Interfacing the GPS Module

Although the gps module works okayish outside, it fails to work accurately when there is something obstructing its connection (eg. Trees, Buildings). For most part the accuracy is close to 3 meters.


## Acceleration

## Power Electronics
