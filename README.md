# Autonomous-2024
This project is a result of a full semester course **ME 691-XIII Autonomous Vehicles** offered at IIT Gandhinagar by prof Harish PM. The project is completely student run and still under progress, there are further developments on the way. We have acheived Level 3 Autonomy on an Electric Golf cart with very less hardware changes.

There are multiple sections of the cart on which modifications were done. ![14-seater-electric-golf-cart](https://github.com/user-attachments/assets/39bc530b-ad84-4064-8a0a-6ae69f878818)

## Perception

### Hardware
### Software

## Steering Control

## Braking

## Localization
### Hardware
Used an Arduino UNO R3 and a Neo 6mv2 GPS Module. The accuracy of this GPS module is close to 2.5~3.5 m while the GPS is fixed to atleast 5 satellites for trilateration.![image](https://github.com/user-attachments/assets/b914471f-1ac3-40b9-9c89-c2872a840783)

### Software
So our aim is to run the cart autonomously anywhere inside our campus. For that we need to know where is it??
One of the difficulties was to get a map of the campus roads but thanks to OpenStreetMaps - ( an open source maps library) that made our jobs much easier. OpenStreetMaps comes with **osmnx** - a library that u can install directly via pip to your python and start using their functionalities like projection and map inference. ![image](https://github.com/user-attachments/assets/0b734ccb-e23b-4299-ae07-0e699680841a)


First we got the map of interest from their Website( Essentially an xml file that contains maps and its lat,long projections). Simply download the map and next we used a library called **networkx** to convert that map to a Graph of nodes and edges. That helps us in applying any path finding algorithm easily like djikstra or A*. ![image](https://github.com/user-attachments/assets/bddeb112-dd67-4628-8732-1265e47f4673)


Next we have a map of the campus roads visible through a tkinter gui window on which we select the Start and end node and a* algorithm returns a path(Red) that the cart can traverse. Meanwhile we can also show our current location on the map(Blue) by projecting the latitude, longitude values coming from the GPS Module back to the map by **pyproj** library. ![WhatsApp Image 2024-11-15 at 17 50 44(1)](https://github.com/user-attachments/assets/8e77a232-152f-4668-973d-db460994d7e4)



### Interfacing the GPS Module

Although the gps module works okayish outside, it fails to work accurately when there is something obstructing its connection (eg. Trees, Buildings). For most part the accuracy is close to 3 meters.


## Acceleration

## Power Electronics
