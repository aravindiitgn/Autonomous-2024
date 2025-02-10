import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class GPSRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Fetch the GPS data
        try:
            result = os.popen('termux-location').read()
            location_data = json.loads(result)
            latitude = location_data['latitude']
            longitude = location_data['longitude']

            # Response
            response = {
                "latitude": latitude,
                "longitude": longitude
            }

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode())

def run(server_class=HTTPServer, handler_class=GPSRequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}...')
    httpd.serve_forever()

if _name_ == "_main_":
    run()