import gpxpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from haversine import haversine
import folium
import statistics

def analyze_gpx_file(gpx_file_path):
    """
    Comprehensive analysis of a GPX file to identify potential issues
    """
    print(f"Analyzing GPX file: {gpx_file_path}")
    
    # Read the GPX file
    with open(gpx_file_path, 'r') as f:
        gpx = gpxpy.parse(f)
    
    # Basic information
    track_count = len(gpx.tracks)
    total_points = sum(len(segment.points) for track in gpx.tracks for segment in track.segments)
    segments_count = sum(len(track.segments) for track in gpx.tracks)
    
    print(f"Basic Information:")
    print(f"  - Tracks: {track_count}")
    print(f"  - Segments: {segments_count}")
    print(f"  - Total points: {total_points}")
    
    # Extract points
    all_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                all_points.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation if point.elevation else 0,
                    'time': point.time
                })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_points)
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(1, len(df)):
        point1 = (df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'])
        point2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        dist = haversine(point1, point2, unit='m')
        distances.append(dist)
    
    # Add distances to DataFrame
    df['distance_to_prev_m'] = [0] + distances
    df['cumulative_distance_km'] = np.cumsum(df['distance_to_prev_m']) / 1000
    
    # Analyze point distribution and potential issues
    print("\nPoint Distribution Analysis:")
    print(f"  - Total distance: {df['cumulative_distance_km'].iloc[-1]:.2f} km")
    print(f"  - Average distance between points: {statistics.mean(distances):.2f} m")
    print(f"  - Median distance between points: {statistics.median(distances):.2f} m")
    print(f"  - Max distance between points: {max(distances):.2f} m")
    print(f"  - Min distance between points: {min(distances):.2f} m")
    
    # Check for unusually large jumps
    large_jumps = [(i+1, dist) for i, dist in enumerate(distances) if dist > 500]
    if large_jumps:
        print("\nPotential Issues - Large Jumps Detected:")
        for idx, dist in large_jumps:
            print(f"  - Point {idx}: {dist:.2f} m jump from previous point")
            print(f"    Location: {df.iloc[idx]['latitude']}, {df.iloc[idx]['longitude']}")
            
    # Check for GPS jitter (many very small movements)
    jitter_segments = []
    current_segment = []
    
    for i, dist in enumerate(distances):
        if dist < 1:  # Less than 1 meter movement is suspect
            current_segment.append(i+1)
        else:
            if len(current_segment) > 5:  # If we had more than 5 very small movements in a row
                jitter_segments.append(current_segment)
            current_segment = []
    
    if current_segment and len(current_segment) > 5:
        jitter_segments.append(current_segment)
    
    if jitter_segments:
        total_jitter_points = sum(len(seg) for seg in jitter_segments)
        print(f"\nPotential GPS Jitter Detected:")
        print(f"  - {len(jitter_segments)} segments with jitter")
        print(f"  - {total_jitter_points} total points affected ({(total_jitter_points/total_points)*100:.1f}% of track)")
        print(f"  - Removing jitter could reduce track length")
    
    # Time gap analysis (if time data is available)
    if 'time' in df.columns and df['time'].iloc[0] is not None:
        time_diffs = []
        for i in range(1, len(df)):
            if df.iloc[i]['time'] and df.iloc[i-1]['time']:
                diff = (df.iloc[i]['time'] - df.iloc[i-1]['time']).total_seconds()
                time_diffs.append(diff)
        
        if time_diffs:
            large_time_gaps = [(i+1, gap) for i, gap in enumerate(time_diffs) if gap > 300]  # 5+ minute gaps
            if large_time_gaps:
                print("\nPotential Issues - Time Gaps Detected:")
                for idx, gap in large_time_gaps:
                    minutes = gap / 60
                    print(f"  - Point {idx}: {minutes:.1f} minute gap")
                    print(f"    Location: {df.iloc[idx]['latitude']}, {df.iloc[idx]['longitude']}")
    
    # Generate visualization
    create_visualization(df, gpx_file_path)
    
    # Generate statistics with potential fixes
    print("\nPossible Solutions:")
    
    # Try different filtering methods and report results
    filtered_dfs = {
        "Original": df,
        "Remove points < 5m apart": filter_by_distance(df.copy(), 5),
        "Remove points < 10m apart": filter_by_distance(df.copy(), 10),
        "Remove jitter clusters": filter_jitter_clusters(df.copy(), jitter_segments)
    }
    
    print("\nDistance Comparison with Filtering:")
    for name, filtered_df in filtered_dfs.items():
        if len(filtered_df) > 1:
            total_dist = calculate_total_distance(filtered_df)
            point_reduction = (1 - len(filtered_df) / len(df)) * 100
            print(f"  - {name}: {total_dist:.2f} km ({len(filtered_df)} points, {point_reduction:.1f}% reduction)")
    
    # Return the full analysis
    return {
        "data": df,
        "stats": {
            "total_distance": df['cumulative_distance_km'].iloc[-1],
            "total_points": total_points,
            "avg_point_distance": statistics.mean(distances),
            "large_jumps": large_jumps,
            "jitter_segments": jitter_segments
        },
        "filtered_data": filtered_dfs
    }

def filter_by_distance(df, min_distance_meters=5):
    """
    Filter out points that are too close together (likely GPS noise)
    """
    if len(df) <= 1:
        return df
        
    cleaned_points = [0]  # Always keep the first point
    last_included = 0
    
    for i in range(1, len(df)):
        point1 = (df.iloc[last_included]['latitude'], df.iloc[last_included]['longitude'])
        point2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        
        # Convert distance to meters
        dist_m = haversine(point1, point2, unit='m')
        
        if dist_m >= min_distance_meters:
            cleaned_points.append(i)
            last_included = i
    
    # Always include the last point
    if cleaned_points[-1] != len(df) - 1:
        cleaned_points.append(len(df) - 1)
        
    return df.iloc[cleaned_points].reset_index(drop=True)

def filter_jitter_clusters(df, jitter_segments):
    """
    Remove identified GPS jitter clusters
    """
    if not jitter_segments or len(df) <= 1:
        return df
    
    # Get all points to remove
    points_to_remove = set()
    for segment in jitter_segments:
        points_to_remove.update(segment)
    
    # Keep only points not in the jitter segments
    keep_indices = [i for i in range(len(df)) if i not in points_to_remove]
    
    return df.iloc[keep_indices].reset_index(drop=True)

def calculate_total_distance(df):
    """
    Calculate total distance for a DataFrame
    """
    if len(df) <= 1:
        return 0
        
    total_dist = 0
    for i in range(1, len(df)):
        point1 = (df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'])
        point2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        total_dist += haversine(point1, point2)
    
    return total_dist

def create_visualization(df, gpx_file_path):
    """
    Create a visualization to help understand the GPX file
    """
    # Plot the distance distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['distance_to_prev_m'], bins=50, alpha=0.7)
    plt.xlabel('Distance Between Points (meters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances Between Consecutive Points')
    plt.grid(True, alpha=0.3)
    plt.savefig('distance_distribution.png')
    
    # Create an interactive map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Add the route
    points = [(row['latitude'], row['longitude']) for _, row in df.iterrows()]
    folium.PolyLine(points, color='blue', weight=3, opacity=0.7).add_to(m)
    
    # Mark start and end
    folium.Marker(
        [df.iloc[0]['latitude'], df.iloc[0]['longitude']],
        popup='Start',
        icon=folium.Icon(color='green')
    ).add_to(m)
    
    folium.Marker(
        [df.iloc[-1]['latitude'], df.iloc[-1]['longitude']],
        popup='End',
        icon=folium.Icon(color='red')
    ).add_to(m)
    
    # Mark any large jumps
    for i in range(1, len(df)):
        if df.iloc[i]['distance_to_prev_m'] > 500:  # Large jump
            folium.CircleMarker(
                [df.iloc[i]['latitude'], df.iloc[i]['longitude']],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                popup=f"Large jump: {df.iloc[i]['distance_to_prev_m']:.1f}m"
            ).add_to(m)
    
    # Save the map
    map_filename = gpx_file_path.replace('.gpx', '_analysis_map.html')
    m.save(map_filename)
    print(f"\nAnalysis map saved to: {map_filename}")

def fix_gpx_file(gpx_file_path, output_path, filter_method='distance', threshold=5):
    """
    Create a fixed version of the GPX file with common issues addressed
    
    Parameters:
    - gpx_file_path: Path to the original GPX file
    - output_path: Path to save the fixed GPX file
    - filter_method: 'distance' or 'jitter'
    - threshold: For distance filtering, minimum distance between points in meters
    
    Returns:
    - Path to the fixed GPX file
    """
    # Read and analyze the original GPX
    with open(gpx_file_path, 'r') as f:
        gpx = gpxpy.parse(f)
    
    analysis = analyze_gpx_file(gpx_file_path)
    
    # Apply the requested filtering
    if filter_method == 'distance':
        filtered_df = filter_by_distance(analysis['data'], threshold)
    elif filter_method == 'jitter':
        filtered_df = filter_jitter_clusters(analysis['data'], analysis['stats']['jitter_segments'])
    else:
        raise ValueError(f"Unknown filter method: {filter_method}")
    
    # Create a new GPX file
    new_gpx = gpxpy.gpx.GPX()
    
    # Create first track
    track = gpxpy.gpx.GPXTrack()
    new_gpx.tracks.append(track)
    
    # Create first segment in track
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    
    # Add filtered points
    for _, row in filtered_df.iterrows():
        point = gpxpy.gpx.GPXTrackPoint(
            latitude=row['latitude'],
            longitude=row['longitude'],
            elevation=row['elevation'],
            time=row['time'] if 'time' in row and row['time'] is not None else None
        )
        segment.points.append(point)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(new_gpx.to_xml())
    
    print(f"Fixed GPX file saved to: {output_path}")
    print(f"Original points: {len(analysis['data'])}")
    print(f"Filtered points: {len(filtered_df)}")
    print(f"Distance reduction: {analysis['data']['cumulative_distance_km'].iloc[-1] - calculate_total_distance(filtered_df):.2f} km")
    
    return output_path


# analysis = analyze_gpx_file("gpx/TSP_solo.gpx")
# fix_gpx_file("your_route.gpx", "fixed_route.gpx", filter_method='distance', threshold=10)