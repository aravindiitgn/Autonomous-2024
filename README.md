# Autonomous-2024
This project is a result of a full semester course **ME 691-XIII Autonomous Vehicles** offered at IIT Gandhinagar by prof Harish PM. The project is still under progress and there are further developments on the way. We have acheived Level 3 Autonomy on an Electric Golf cart with very less hardware changes.

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



### Hardware
### Software

## Acceleration

## Power Electronics
