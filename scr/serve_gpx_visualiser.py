import http.server
import socketserver
import os
import webbrowser
import json
import urllib.parse
from pathlib import Path

# Configuration
PORT = 8000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 'scr/' directory
GPX_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '../gpx/'))  # '../gpx/' directory

class GPXHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SCRIPT_DIR, **kwargs)
    
    def do_GET(self):
        # Special route to list available GPX files
        if self.path == '/list_gpx_files':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # CORS header for local dev
            self.end_headers()
            
            # Get list of GPX files
            gpx_files = []
            if os.path.exists(GPX_DIR):
                gpx_files = [f for f in os.listdir(GPX_DIR) if f.endswith('.gpx')]
            
            # Send file list as JSON
            self.wfile.write(json.dumps(gpx_files).encode())
            return
        
        # Route to serve GPX file content
        elif self.path.startswith('/gpx/'):
            # Important change: decode URL-encoded filename
            encoded_filename = self.path.split('/')[-1]
            filename = urllib.parse.unquote(encoded_filename)
            
            file_path = os.path.join(GPX_DIR, filename)
            
            print(f"Requested GPX file: {filename}")
            print(f"Looking for file at: {file_path}")
            
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self.send_response(200)
                self.send_header('Content-type', 'application/gpx+xml')
                self.end_headers()
                
                with open(file_path, 'rb') as file:
                    self.wfile.write(file.read())
                return
            else:
                self.send_error(404, f"File not found: {filename}")
                return
        
        # Default behavior for all other routes
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def main():
    print(f"Starting server at http://localhost:{PORT}")
    print(f"Serving files from: {SCRIPT_DIR}")
    print(f"Looking for GPX files in: {GPX_DIR}")
    
    # Check if the GPX directory exists
    if not os.path.exists(GPX_DIR):
        print(f"Warning: GPX directory '{GPX_DIR}' does not exist. Creating it...")
        os.makedirs(GPX_DIR)
    
    # Count GPX files
    gpx_files = [f for f in os.listdir(GPX_DIR) if f.endswith('.gpx')]
    print(f"Found {len(gpx_files)} GPX files: {', '.join(gpx_files)}")
    
    # Open the browser automatically
    url = f"http://localhost:{PORT}/gpx_visualiser.html"
    webbrowser.open(url)
    
    # Start the server
    with socketserver.TCPServer(("", PORT), GPXHandler) as httpd:
        print("Server started. Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()