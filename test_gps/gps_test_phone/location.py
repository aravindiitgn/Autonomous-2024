import requests

# Replace phone_ip with the actual IP address of your phone
try:
    response = requests.get('http://10.7.52.167:8080')
    if response.status_code == 200:
        location_data = response.json()
        print(f"Latitude: {location_data['latitude']}, Longitude: {location_data['longitude']}")
    else:
        print("Failed to retrieve data:", response.status_code)
except Exception as e:
    print("Error accessing server:", e)