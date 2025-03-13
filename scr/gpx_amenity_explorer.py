import os
import json
import gpxpy
import folium
import requests
from flask import Flask, render_template_string, request, jsonify
from pathlib import Path
from threading import Timer
import webbrowser

# HTML template as a string - this avoids needing a templates directory
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>GPX Route Amenity Explorer</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <style>
        body { margin: 0; padding: 0; }
        #map { position: absolute; top: 0; bottom: 0; right: 0; width: 75%; height: 100%; }
        #sidebar { position: absolute; top: 0; bottom: 0; left: 0; width: 25%; height: 100%; background: #f8f9fa; padding: 10px; overflow-y: auto; }
        h3 { margin-top: 0; }
        .route-item, .amenity-item { 
            cursor: pointer; 
            padding: 8px; 
            margin-bottom: 5px; 
            border-radius: 4px;
        }
        .route-item:hover { background-color: #e9ecef; }
        .route-item.selected { background-color: #cfe2ff; }
        .amenity-section { margin-top: 20px; padding-top: 10px; border-top: 1px solid #dee2e6; }
        .amenity-count { font-size: 0.8em; color: #6c757d; margin-left: 5px; }
        .loading { margin-top: 10px; display: none; }
        .legend { padding: 6px 8px; background: white; background: rgba(255,255,255,0.8); border-radius: 5px; }
        .legend i { width: 18px; height: 18px; float: left; margin-right: 8px; opacity: 0.7; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h3>GPX Routes</h3>
        <div id="routes-list"></div>
        
        <div class="amenity-section">
            <h3>Amenities</h3>
            <p>Select a route to view nearby amenities</p>
            <div id="amenities-list"></div>
            <div class="loading">Loading amenities...</div>
        </div>
    </div>
    <div id="map">{{ map|safe }}</div>

    <script>
        // Parse the routes data
        const routes = {{ routes|safe }};
        
        // Initialize map (replacing the folium map)
        const map = L.map('map').setView([0, 0], 2);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);
        
        // Variables to store map layers
        const routeLayers = {};
        let selectedRouteId = null;
        let amenityMarkers = L.layerGroup().addTo(map);
        
        // Amenity types and colors
        const amenityColors = {
            'restaurant': '#FF5733',
            'cafe': '#C70039',
            'bar': '#900C3F',
            'pub': '#581845',
            'fast_food': '#FFC300',
            'water_point': '#33A1FF',
            'drinking_water': '#33A1FF',
            'toilets': '#DAF7A6',
            'bench': '#5D6D7E',
            'shelter': '#7D3C98',
            'bicycle_rental': '#2E86C1',
            'bicycle_repair_station': '#2E86C1',
            'fuel': '#E74C3C',
            'atm': '#F1C40F',
            'bank': '#F1C40F',
            'pharmacy': '#27AE60',
            'hospital': '#C0392B',
            'clinic': '#E67E22',
            'shop': '#8E44AD',
            'supermarket': '#16A085',
            'convenience': '#D35400',
            'bakery': '#7D6608'
        };
        
        // Initialize amenity types checkboxes
        const amenityTypes = {{ amenity_types|tojson }};
        const selectedAmenities = new Set();
        
        // Function to render routes list
        function renderRoutesList() {
            const routesList = document.getElementById('routes-list');
            routesList.innerHTML = '';
            
            routes.forEach(route => {
                const routeItem = document.createElement('div');
                routeItem.className = 'route-item';
                routeItem.textContent = route.name;
                routeItem.dataset.id = route.id;
                
                if (route.id === selectedRouteId) {
                    routeItem.classList.add('selected');
                }
                
                routeItem.addEventListener('click', () => selectRoute(route.id));
                routesList.appendChild(routeItem);
                
                // Add route to map
                if (!routeLayers[route.id]) {
                    const routeCoords = route.points.map(point => [point[0], point[1]]);
                    const routeLayer = L.polyline(routeCoords, {
                        color: '#3388ff',
                        weight: 4,
                        opacity: 0.6
                    }).addTo(map);
                    
                    routeLayer.on('click', () => selectRoute(route.id));
                    routeLayers[route.id] = routeLayer;
                }
            });
            
            // Fit map to show all routes
            if (Object.keys(routeLayers).length > 0) {
                const bounds = [];
                Object.values(routeLayers).forEach(layer => {
                    bounds.push(...layer.getLatLngs());
                });
                map.fitBounds(bounds);
            }
        }
        
        // Function to render amenity checkboxes
        function renderAmenitiesList() {
            const amenitiesList = document.getElementById('amenities-list');
            
            if (!selectedRouteId) {
                amenitiesList.innerHTML = '<p>Select a route first</p>';
                return;
            }
            
            amenitiesList.innerHTML = '';
            
            amenityTypes.forEach(type => {
                const amenityItem = document.createElement('div');
                amenityItem.className = 'amenity-item';
                
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `amenity-${type}`;
                checkbox.checked = selectedAmenities.has(type);
                checkbox.addEventListener('change', () => {
                    if (checkbox.checked) {
                        selectedAmenities.add(type);
                    } else {
                        selectedAmenities.delete(type);
                    }
                    fetchAmenities();
                });
                
                const label = document.createElement('label');
                label.htmlFor = `amenity-${type}`;
                label.textContent = type.replace('_', ' ');
                
                const countSpan = document.createElement('span');
                countSpan.className = 'amenity-count';
                countSpan.id = `count-${type}`;
                
                amenityItem.appendChild(checkbox);
                amenityItem.appendChild(label);
                amenityItem.appendChild(countSpan);
                amenitiesList.appendChild(amenityItem);
            });
        }
        
        // Function to select a route
        function selectRoute(routeId) {
            // Reset previous selection
            if (selectedRouteId && routeLayers[selectedRouteId]) {
                routeLayers[selectedRouteId].setStyle({
                    color: '#3388ff',
                    weight: 4,
                    opacity: 0.6
                });
            }
            
            // Set new selection
            selectedRouteId = routeId;
            
            if (routeLayers[selectedRouteId]) {
                routeLayers[selectedRouteId].setStyle({
                    color: '#ff3333',
                    weight: 6,
                    opacity: 0.8
                });
                
                // Zoom to the selected route
                map.fitBounds(routeLayers[selectedRouteId].getBounds());
            }
            
            // Update UI
            renderRoutesList();
            renderAmenitiesList();
            
            // Clear existing amenity markers
            amenityMarkers.clearLayers();
            
            // If there are selected amenities, fetch them
            if (selectedAmenities.size > 0) {
                fetchAmenities();
            }
        }
        
        // Function to fetch amenities for the selected route
        function fetchAmenities() {
            if (!selectedRouteId || selectedAmenities.size === 0) {
                amenityMarkers.clearLayers();
                return;
            }
            
            const loadingEl = document.querySelector('.loading');
            loadingEl.style.display = 'block';
            
            // Clear count indicators
            amenityTypes.forEach(type => {
                const countEl = document.getElementById(`count-${type}`);
                if (countEl) countEl.textContent = '';
            });
            
            fetch('/get_amenities', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    route_id: selectedRouteId,
                    amenities: Array.from(selectedAmenities)
                })
            })
            .then(response => response.json())
            .then(data => {
                amenityMarkers.clearLayers();
                
                // Group amenities by type for counting
                const typeCounts = {};
                
                // Add markers for each amenity
                data.amenities.forEach(amenity => {
                    // Update count for this type
                    typeCounts[amenity.type] = (typeCounts[amenity.type] || 0) + 1;
                    
                    // Skip if this amenity type is not selected (shouldn't happen, but just in case)
                    if (!selectedAmenities.has(amenity.type)) return;
                    
                    const color = amenityColors[amenity.type] || '#3388ff';
                    
                    const markerIcon = L.divIcon({
                        html: `<div style="background-color: ${color}; width: 10px; height: 10px; border-radius: 50%;"></div>`,
                        className: 'amenity-marker',
                        iconSize: [10, 10]
                    });
                    
                    const marker = L.marker([amenity.lat, amenity.lon], {
                        icon: markerIcon
                    });
                    
                    marker.bindPopup(`
                        <strong>${amenity.name || amenity.type}</strong><br>
                        Type: ${amenity.type}
                    `);
                    
                    amenityMarkers.addLayer(marker);
                });
                
                // Update the count indicators
                Object.entries(typeCounts).forEach(([type, count]) => {
                    const countEl = document.getElementById(`count-${type}`);
                    if (countEl) countEl.textContent = `(${count})`;
                });
                
                // Add a legend to the map
                updateLegend();
            })
            .catch(error => {
                console.error('Error fetching amenities:', error);
            })
            .finally(() => {
                loadingEl.style.display = 'none';
            });
        }
        
        // Function to update the map legend
        function updateLegend() {
            // Remove existing legend if any
            const existingLegend = document.querySelector('.legend');
            if (existingLegend) {
                existingLegend.remove();
            }
            
            if (selectedAmenities.size === 0) return;
            
            // Create new legend
            const legend = L.control({ position: 'bottomright' });
            
            legend.onAdd = function (map) {
                const div = L.DomUtil.create('div', 'legend');
                div.innerHTML = '<h4>Amenities</h4>';
                
                Array.from(selectedAmenities).forEach(type => {
                    const color = amenityColors[type] || '#3388ff';
                    div.innerHTML += `
                        <div><i style="background:${color}"></i>${type.replace('_', ' ')}</div>
                    `;
                });
                
                return div;
            };
            
            legend.addTo(map);
        }
        
        // Initialize the UI
        renderRoutesList();
        renderAmenitiesList();
    </script>
</body>
</html>
"""

app = Flask(__name__)

# Store the processed GPX data
routes = []

def parse_gpx_files(directory):
    """Parse all GPX files in the given directory."""
    gpx_files = list(Path(directory).glob('*.gpx'))
    
    for idx, gpx_file in enumerate(gpx_files):
        with open(gpx_file, 'r') as f:
            gpx = gpxpy.parse(f)
            
        for track_idx, track in enumerate(gpx.tracks):
            track_points = []
            for segment in track.segments:
                for point in segment.points:
                    track_points.append((point.latitude, point.longitude))
            
            if track_points:
                route_name = track.name if track.name else f"{gpx_file.stem} - Track {track_idx+1}"
                routes.append({
                    'id': f"route_{idx}_{track_idx}",
                    'name': route_name,
                    'points': track_points
                })
    
    return routes

def get_route_bounds(route_points):
    """Calculate the bounding box for a route."""
    lats = [p[0] for p in route_points]
    lons = [p[1] for p in route_points]
    return [
        [min(lats), min(lons)],
        [max(lats), max(lons)]
    ]

def query_overpass_for_amenities(route_points, amenity_types, buffer_meters=200):
    """Query the Overpass API for amenities along a route."""
    # Simplify the route for the query (take every 10th point to reduce query size)
    simplified_points = route_points[::10]
    
    # Build a Overpass query to find amenities along the route within buffer distance
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # For each point, we'll create a circle with the given buffer radius
    # and look for amenities within these circles
    query_parts = []
    for lat, lon in simplified_points:
        query_parts.append(f"node(around:{buffer_meters},{lat},{lon})[amenity~\"^({amenity_types})$\"];")
    
    overpass_query = f"""
    [out:json];
    (
        {''.join(query_parts)}
    );
    out body;
    """
    
    response = requests.post(overpass_url, data={"data": overpass_query})
    
    if response.status_code != 200:
        return []
    
    data = response.json()
    results = []
    
    for element in data.get('elements', []):
        if element.get('type') == 'node':
            amenity_type = element.get('tags', {}).get('amenity')
            results.append({
                'id': element.get('id'),
                'type': amenity_type,
                'name': element.get('tags', {}).get('name', f"{amenity_type}"),
                'lat': element.get('lat'),
                'lon': element.get('lon')
            })
    
    return results

@app.route('/')
def index():
    # Create a Folium map centered at the first route's first point if available
    center_lat, center_lon = 0, 0
    zoom_start = 10
    
    if routes:
        first_route = routes[0]
        center_lat, center_lon = first_route['points'][0]
        
        # Calculate bounds for all routes to set appropriate zoom
        all_lats = [p[0] for route in routes for p in route['points']]
        all_lons = [p[1] for route in routes for p in route['points']]
        
        # Adjust zoom_start based on the range of coordinates
        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        zoom_start = 12  # Default
        if max(lat_range, lon_range) > 0.5:
            zoom_start = 9
        elif max(lat_range, lon_range) < 0.05:
            zoom_start = 14
    
    map_obj = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, 
                        control_scale=True)
    
    # Convert route data to JSON for front-end use
    routes_json = json.dumps(routes)
    
    # Add common amenity types
    amenity_types = [
        "restaurant", "cafe", "fast_food",
        "water_point", "drinking_water", "toilets", 
        "fuel", "atm", "bank", "pharmacy", "hospital", 
        "shop", "supermarket", "convenience", "bakery"
    ]
    
    # Render the template directly from the string
    return render_template_string(HTML_TEMPLATE,
                                 map=map_obj._repr_html_(),
                                 routes=routes_json,
                                 amenity_types=amenity_types)

@app.route('/get_amenities', methods=['POST'])
def get_amenities():
    data = request.json
    route_id = data.get('route_id')
    selected_amenities = data.get('amenities', [])
    
    if not route_id or not selected_amenities:
        return jsonify({'error': 'Missing parameters'}), 400
    
    # Find the selected route
    selected_route = next((r for r in routes if r['id'] == route_id), None)
    if not selected_route:
        return jsonify({'error': 'Route not found'}), 404
    
    # Join amenity types for the Overpass query
    amenity_types_str = '|'.join(selected_amenities)
    
    # Query Overpass API
    amenities = query_overpass_for_amenities(
        selected_route['points'], 
        amenity_types_str
    )
    
    return jsonify({'amenities': amenities})

def main():
    """Main function to run the application."""
    # Ask for the directory containing GPX files
    gpx_dir = input("Enter the directory path containing your GPX files: ")
    
    if not os.path.isdir(gpx_dir):
        print(f"Error: Directory '{gpx_dir}' does not exist.")
        return
    
    # Parse GPX files
    print(f"Parsing GPX files in '{gpx_dir}'...")
    parse_gpx_files(gpx_dir)
    
    if not routes:
        print("No valid GPX routes found in the directory.")
        return
    
    print(f"Found {len(routes)} routes.")
    
    # Start the Flask app
    host = '127.0.0.1'
    port = 5000
    
    # Open browser automatically
    def open_browser():
        webbrowser.open(f'http://{host}:{port}/')
    
    Timer(1, open_browser).start()
    
    # Run the Flask app
    print(f"Starting the server at http://{host}:{port}/")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()