import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def parse_gpx(gpx_file):
    """Parse GPX file and extract track points."""
    # Parse the XML
    tree = ET.parse(gpx_file)
    root = tree.getroot()
    
    # GPX namespace
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    
    # Extract track points
    track_points = []
    
    # Try to find tracks (most common)
    tracks = root.findall('.//gpx:trk', ns)
    if tracks:
        for track in tracks:
            segments = track.findall('.//gpx:trkseg', ns)
            for segment in segments:
                points = segment.findall('.//gpx:trkpt', ns)
                for point in points:
                    lat = float(point.get('lat'))
                    lon = float(point.get('lon'))
                    track_points.append((lat, lon))
    
    # If no tracks found, try routes
    if not track_points:
        routes = root.findall('.//gpx:rte', ns)
        for route in routes:
            points = route.findall('.//gpx:rtept', ns)
            for point in points:
                lat = float(point.get('lat'))
                lon = float(point.get('lon'))
                track_points.append((lat, lon))
    
    # If no routes found, try waypoints
    if not track_points:
        waypoints = root.findall('.//gpx:wpt', ns)
        for point in waypoints:
            lat = float(point.get('lat'))
            lon = float(point.get('lon'))
            track_points.append((lat, lon))
    
    return track_points

def create_route_visualization(track_points, output_file=None, line_color='black', 
                              background_color='white', line_width=2, dpi=300):
    """Create a minimalist visualization of the route."""
    if not track_points:
        print("No track points found in the GPX file.")
        return
    
    # Extract latitudes and longitudes
    lats = [point[0] for point in track_points]
    lons = [point[1] for point in track_points]
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=background_color)
    
    # Plot the route
    ax.plot(lons, lats, color=line_color, linewidth=line_width, solid_capstyle='round')
    
    # Mark start and end points
    ax.plot(lons[0], lats[0], 'go', markersize=5)  # Start point in green
    ax.plot(lons[-1], lats[-1], 'ro', markersize=5)  # End point in red
    
    # Clean up the plot - remove axis, ticks, etc. for minimalist look
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add a larger margin around the route to ensure endpoints are visible
    margin = 0.005  # Increased margin
    ax.set_xlim(min(lons) - margin, max(lons) + margin)
    ax.set_ylim(min(lats) - margin, max(lats) + margin)
    
    # Save the figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        print(f"Route visualization saved to {output_file}")
    
    plt.tight_layout()
    plt.show()

def main():
    # Check if a file path was provided
    if len(sys.argv) < 2:
        print("Usage: python gpx_visualizer.py <path_to_gpx_file> [output_image_path]")
        sys.exit(1)
    
    gpx_file = sys.argv[1]
    
    # Check if output file is specified
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Auto-generate output filename
        input_path = Path(gpx_file)
        output_file = str(input_path.with_suffix('.png'))
    
    try:
        # Parse GPX file
        track_points = parse_gpx(gpx_file)
        
        # Create visualization
        create_route_visualization(
            track_points=track_points,
            output_file=output_file,
            line_color='black',  # Black line
            background_color='white',
            line_width=2,
            dpi=300
        )
    except Exception as e:
        print(f"Error processing GPX file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()