import gpxpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine
import os
from collections import defaultdict
import math
import requests
import time
import folium
from folium.plugins import HeatMap
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class GPXRouteAnalyzer:
    """
    A class for analyzing and comparing GPX route files from Los Angeles to Las Vegas
    with specific focus on 125km segment summaries and terrain analysis using Google Maps API
    """
    
    def __init__(self, segment_length=125, google_maps_api_key=None):
        """
        Initialize the analyzer with a specified segment length in kilometers
        
        Args:
            segment_length (float): Length of segments for summary in kilometers (default 125km)
            google_maps_api_key (str): Google Maps API key for elevation and terrain data
        """
        self.routes = {}
        self.segment_length = segment_length
        self.segment_summaries = {}
        self.route_metrics = {}
        self.terrain_data = {}
        self.google_maps_api_key = google_maps_api_key
        
    def load_gpx_file(self, file_path, route_name=None):
        """
        Load and parse a GPX file
        
        Args:
            file_path (str): Path to the GPX file
            route_name (str, optional): Name for the route. If None, uses the filename
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if route_name is None:
            route_name = os.path.basename(file_path).split('.')[0]
            
        try:
            with open(file_path, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)
                
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        points.append({
                            'latitude': point.latitude,
                            'longitude': point.longitude,
                            'elevation': point.elevation,
                            'time': point.time
                        })
            
            if not points:
                print(f"No track points found in {file_path}")
                return False
                
            self.routes[route_name] = pd.DataFrame(points)
            print(f"Loaded {len(points)} points from {file_path}")
            
            # Process the route data
            self._process_route(route_name)
            
            # Fetch terrain data if API key is provided
            if self.google_maps_api_key:
                self._fetch_terrain_data(route_name)
                
            return True
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return False
    
    def _process_route(self, route_name):
        """
        Process a route to calculate distance, elevation changes, and gradients
        
        Args:
            route_name (str): Name of the route to process
        """
        if route_name not in self.routes:
            print(f"Route {route_name} not found")
            return
            
        df = self.routes[route_name]
        
        # Calculate distances between consecutive points
        distances = []
        for i in range(len(df) - 1):
            point1 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
            point2 = (df.iloc[i+1]['latitude'], df.iloc[i+1]['longitude'])
            distance = haversine(point1, point2)
            distances.append(distance)
        distances.append(0)  # For the last point
        
        df['distance_km'] = distances
        df['cumulative_distance'] = np.cumsum(df['distance_km'])
        
        # Calculate elevation changes and filter out unrealistic jumps
        elevation_diffs = []
        prev_elevation = df.iloc[0]['elevation']
        
        for i in range(len(df) - 1):
            elev1 = prev_elevation
            elev2 = df.iloc[i+1]['elevation']
            
            # Check for unrealistic elevation changes (more than 100m per 100m horizontal)
            distance_m = df.iloc[i]['distance_km'] * 1000
            max_reasonable_change = min(distance_m, 100) * 0.7  # 70% gradient max
            
            if distance_m > 0 and abs(elev2 - elev1) > max_reasonable_change:
                # Limit the elevation change to something realistic
                if elev2 > elev1:
                    elev_diff = max_reasonable_change
                else:
                    elev_diff = -max_reasonable_change
                print(f"Warning: Limiting unrealistic elevation change at point {i} from {elev2-elev1}m to {elev_diff}m")
            else:
                elev_diff = elev2 - elev1
                
            elevation_diffs.append(elev_diff)
            prev_elevation = elev1 + elev_diff
            
        elevation_diffs.append(0)  # For the last point
        
        df['elevation_diff'] = elevation_diffs
        
        # Smooth elevation data to reduce noise (optional)
        window_size = 5  # Adjust as needed
        df['elevation_smooth'] = df['elevation'].rolling(window=window_size, center=True).mean()
        df['elevation_smooth'] = df['elevation_smooth'].fillna(df['elevation'])
        
        # Recalculate elevation differences after smoothing
        elevation_diffs_smooth = []
        for i in range(len(df) - 1):
            elev1 = df.iloc[i]['elevation_smooth']
            elev2 = df.iloc[i+1]['elevation_smooth']
            elevation_diffs_smooth.append(elev2 - elev1)
        elevation_diffs_smooth.append(0)  # For the last point
        
        df['elevation_diff_smooth'] = elevation_diffs_smooth
        
        # Calculate gradients using smoothed elevation data
        gradients = []
        for i in range(len(df) - 1):
            distance = df.iloc[i]['distance_km'] * 1000  # Convert to meters
            if distance > 0:
                # Use smoothed elevation differences for more realistic gradients
                gradient = (df.iloc[i]['elevation_diff_smooth'] / distance) * 100
                # Apply reasonable limits to gradient calculations (typically -45% to +45%)
                gradient = max(-45, min(45, gradient))
                gradients.append(gradient)
            else:
                gradients.append(0)
        gradients.append(0)  # For the last point
        
        df['gradient'] = gradients
        
        # Apply a rolling mean to further smooth gradients
        df['gradient_smooth'] = df['gradient'].rolling(window=10, center=True).mean().fillna(df['gradient'])
        
        # Calculate segment summaries
        self._calculate_segment_summaries(route_name)
        
        # Calculate overall route metrics
        self._calculate_route_metrics(route_name)
    
    def _fetch_terrain_data(self, route_name):
        """
        Fetch terrain data for a route using Google Maps API
        
        Args:
            route_name (str): Name of the route
        """
        df = self.routes[route_name]
        
        # We'll sample points to avoid exceeding API limits and costs
        # For a 450km route, sampling every 5km gives ~90 API calls
        sample_distance = 5  # km
        
        terrain_data = []
        terrain_types = set()
        last_cumulative_dist = 0
        
        for i, row in df.iterrows():
            current_dist = row['cumulative_distance']
            
            # Only sample at specified intervals
            if current_dist - last_cumulative_dist >= sample_distance or i == 0 or i == len(df) - 1:
                lat, lon = row['latitude'], row['longitude']
                
                # Get terrain type using Places API
                url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=100&key={self.google_maps_api_key}"
                
                try:
                    response = requests.get(url)
                    data = response.json()
                    
                    # Extract terrain types from the types of nearby places and features
                    nearby_types = []
                    if 'results' in data:
                        for result in data['results']:
                            if 'types' in result:
                                nearby_types.extend(result['types'])
                    
                    # Classify terrain based on nearby features
                    terrain_type = self._classify_terrain(nearby_types, lat, lon)
                    terrain_types.add(terrain_type)
                    
                    terrain_data.append({
                        'latitude': lat,
                        'longitude': lon,
                        'distance': current_dist,
                        'terrain_type': terrain_type
                    })
                    
                    last_cumulative_dist = current_dist
                    
                    # Respect API rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error fetching terrain data: {e}")
        
        self.terrain_data[route_name] = pd.DataFrame(terrain_data)
        
        # Add terrain data to the route metrics
        if route_name in self.route_metrics and terrain_data:
            terrain_counts = {t: 0 for t in terrain_types}
            for item in terrain_data:
                terrain_counts[item['terrain_type']] += 1
                
            # Calculate terrain distribution
            total_points = len(terrain_data)
            terrain_distribution = {t: count / total_points for t, count in terrain_counts.items()}
            
            # Add terrain distribution to route metrics
            self.route_metrics[route_name]['terrain_distribution'] = terrain_distribution
            
            # Calculate terrain difficulty score (simplified)
            terrain_difficulty = self._calculate_terrain_difficulty(terrain_distribution)
            self.route_metrics[route_name]['terrain_difficulty'] = terrain_difficulty
            
            # Update the overall difficulty score to include terrain
            self.route_metrics[route_name]['difficulty_score'] = (
                self.route_metrics[route_name]['difficulty_score'] * 0.8 +  # 80% of original score
                terrain_difficulty * 20  # 20% terrain contribution (0-20 points)
            )
    
    def _classify_terrain(self, nearby_types, lat, lon):
        """
        Classify terrain based on nearby place types and geographic location
        
        Args:
            nearby_types (list): List of place types from Google Maps API
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            str: Classified terrain type
        """
        # First check specific terrain indicators
        if any(t in nearby_types for t in ['natural_feature', 'park']):
            return 'natural'
        elif any(t in nearby_types for t in ['route', 'street_address', 'road']):
            return 'road'
        elif any(t in nearby_types for t in ['locality', 'sublocality', 'neighborhood']):
            return 'urban'
            
        # Fallback to Elevation API for terrain classification based on elevation change
        # in surrounding area
        radius = 500  # meters
        points = [
            f"{lat},{lon}",  # Center
            f"{lat+0.003},{lon}",  # North
            f"{lat-0.003},{lon}",  # South
            f"{lat},{lon+0.003}",  # East
            f"{lat},{lon-0.003}"   # West
        ]
        locations = "|".join(points)
        
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={self.google_maps_api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'results' in data:
                elevations = [result['elevation'] for result in data['results']]
                
                # Calculate elevation variance as a measure of terrain roughness
                if len(elevations) > 1:
                    center_elev = elevations[0]
                    elev_diffs = [abs(e - center_elev) for e in elevations[1:]]
                    avg_diff = sum(elev_diffs) / len(elev_diffs)
                    
                    if avg_diff > 20:
                        return 'mountainous'
                    elif avg_diff > 5:
                        return 'hilly'
                    else:
                        return 'flat'
        
        except Exception as e:
            print(f"Error classifying terrain: {e}")
            
        # Default if all else fails
        return 'unknown'
    
    def _calculate_terrain_difficulty(self, terrain_distribution):
        """
        Calculate terrain difficulty score based on terrain types
        
        Args:
            terrain_distribution (dict): Distribution of terrain types
            
        Returns:
            float: Terrain difficulty score (0-100)
        """
        # Terrain difficulty weights (0-100 scale)
        difficulty_weights = {
            'flat': 20,
            'road': 30,
            'hilly': 50,
            'urban': 60,
            'natural': 70,
            'mountainous': 90,
            'unknown': 50
        }
        
        # Calculate weighted average
        difficulty = 0
        for terrain_type, proportion in terrain_distribution.items():
            difficulty += difficulty_weights.get(terrain_type, 50) * proportion
            
        return difficulty
    
    def _calculate_segment_summaries(self, route_name):
        """
        Calculate summaries for each segment of the specified length
        
        Args:
            route_name (str): Name of the route
        """
        df = self.routes[route_name]
        total_distance = df['cumulative_distance'].max()
        num_segments = math.ceil(total_distance / self.segment_length)
        
        summaries = []
        
        for segment in range(num_segments):
            start_dist = segment * self.segment_length
            end_dist = min((segment + 1) * self.segment_length, total_distance)
            
            # Get points in this segment
            segment_df = df[(df['cumulative_distance'] >= start_dist) & 
                            (df['cumulative_distance'] <= end_dist)]
            
            if segment_df.empty:
                continue
                
            # Calculate segment metrics
            segment_distance = end_dist - start_dist
            
            # Elevation data - use smoothed elevation for more reliable metrics
            elevation_gain = segment_df[segment_df['elevation_diff_smooth'] > 0]['elevation_diff_smooth'].sum()
            elevation_loss = abs(segment_df[segment_df['elevation_diff_smooth'] < 0]['elevation_diff_smooth'].sum())
            max_elevation = segment_df['elevation_smooth'].max()
            min_elevation = segment_df['elevation_smooth'].min()
            
            # Gradient data - use smoothed gradients
            avg_gradient = segment_df['gradient_smooth'].mean()
            max_gradient = segment_df['gradient_smooth'].max()
            steep_sections = len(segment_df[segment_df['gradient_smooth'] > 8])  # Count sections steeper than 8%
            
            # Store summary
            summaries.append({
                'segment': segment + 1,
                'start_km': start_dist,
                'end_km': end_dist,
                'distance_km': segment_distance,
                'elevation_gain_m': elevation_gain,
                'elevation_loss_m': elevation_loss,
                'max_elevation_m': max_elevation,
                'min_elevation_m': min_elevation,
                'avg_gradient_percent': avg_gradient,
                'max_gradient_percent': max_gradient,
                'steep_sections': steep_sections,
                'elevation_gain_per_km': elevation_gain / segment_distance if segment_distance > 0 else 0
            })
            
        self.segment_summaries[route_name] = pd.DataFrame(summaries)
    
    def _calculate_route_metrics(self, route_name):
        """
        Calculate overall metrics for the route
        
        Args:
            route_name (str): Name of the route
        """
        df = self.routes[route_name]
        
        total_distance = df['cumulative_distance'].max()
        
        # Use smoothed elevation data for more reliable metrics
        elevation_gain = df[df['elevation_diff_smooth'] > 0]['elevation_diff_smooth'].sum()
        elevation_loss = abs(df[df['elevation_diff_smooth'] < 0]['elevation_diff_smooth'].sum())
        max_elevation = df['elevation_smooth'].max()
        min_elevation = df['elevation_smooth'].min()
        
        # Use smoothed gradient data
        avg_gradient = df['gradient_smooth'].mean()
        max_gradient = df['gradient_smooth'].max()
        
        # Calculate steep sections (>8% gradient)
        steep_sections = len(df[df['gradient_smooth'] > 8])
        steep_percentage = (steep_sections / len(df)) * 100 if len(df) > 0 else 0
        
        # Calculate a terrain variability index based on gradient changes
        gradient_changes = np.abs(np.diff(df['gradient_smooth']))
        terrain_variability = np.mean(gradient_changes)
        
        # Calculate a difficulty score (simplified example)
        distance_score = min(total_distance / 450 * 25, 25)  # Max 25 points for distance
        elevation_score = min(elevation_gain / 5000 * 35, 35)  # Max 35 points for elevation gain
        gradient_score = min(max_gradient / 20 * 25, 25)  # Max 25 points for max gradient
        terrain_score = min(terrain_variability / 5 * 15, 15)  # Max 15 points for terrain variability
        
        difficulty_score = distance_score + elevation_score + gradient_score + terrain_score
        
        self.route_metrics[route_name] = {
            'total_distance_km': total_distance,
            'elevation_gain_m': elevation_gain,
            'elevation_loss_m': elevation_loss,
            'max_elevation_m': max_elevation,
            'min_elevation_m': min_elevation,
            'avg_gradient_percent': avg_gradient,
            'max_gradient_percent': max_gradient,
            'steep_sections_count': steep_sections,
            'steep_sections_percent': steep_percentage,
            'terrain_variability_index': terrain_variability,
            'difficulty_score': difficulty_score
        }
    
    def get_segment_summary(self, route_name):
        """
        Get segment summaries for a route
        
        Args:
            route_name (str): Name of the route
            
        Returns:
            pd.DataFrame: Segment summaries or None if not found
        """
        return self.segment_summaries.get(route_name)
    
    def get_route_metrics(self, route_name=None):
        """
        Get metrics for all routes or a specific route
        
        Args:
            route_name (str, optional): Name of the route. If None, returns all routes
            
        Returns:
            pd.DataFrame or dict: Route metrics
        """
        if route_name:
            return self.route_metrics.get(route_name)
        
        # Convert all metrics to a DataFrame for comparison
        metrics_df = pd.DataFrame()
        
        for name, metrics in self.route_metrics.items():
            # Handle the terrain distribution separately
            terrain_dist = {}
            if 'terrain_distribution' in metrics:
                terrain_dist = metrics.pop('terrain_distribution')
                
            # Add the main metrics
            route_df = pd.DataFrame([metrics], index=[name])
            
            # Add terrain distribution as separate columns
            if terrain_dist:
                for terrain, value in terrain_dist.items():
                    route_df[f'terrain_{terrain}_percent'] = value * 100
                    
            if metrics_df.empty:
                metrics_df = route_df
            else:
                metrics_df = pd.concat([metrics_df, route_df])
                
        return metrics_df
    
    def plot_elevation_profiles(self, route_names=None, figsize=(12, 6), highlight_segments=True):
        """
        Plot elevation profiles for the specified routes with segment markers
        
        Args:
            route_names (list, optional): Names of routes to plot. If None, plots all routes
            figsize (tuple, optional): Figure size
            highlight_segments (bool): Whether to highlight 125km segments
        """
        if route_names is None:
            route_names = list(self.routes.keys())
            
        plt.figure(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(route_names)))
        
        for i, route_name in enumerate(route_names):
            if route_name not in self.routes:
                print(f"Route {route_name} not found")
                continue
                
            df = self.routes[route_name]
            plt.plot(df['cumulative_distance'], df['elevation'], 
                    label=route_name, color=colors[i])
            
            # Add segment markers if requested
            if highlight_segments and route_name in self.segment_summaries:
                segments = self.segment_summaries[route_name]
                for _, segment in segments.iterrows():
                    plt.axvline(x=segment['start_km'], color=colors[i], linestyle='--', alpha=0.5)
                    plt.text(segment['start_km'], plt.ylim()[1]*0.9, 
                            f"Seg {segment['segment']}", rotation=90, color=colors[i])
            
        plt.xlabel('Distance (km)')
        plt.ylabel('Elevation (m)')
        plt.title('Elevation Profiles')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_profiles(self, route_names=None, figsize=(12, 6)):
        """
        Plot gradient profiles for the specified routes
        
        Args:
            route_names (list, optional): Names of routes to plot. If None, plots all routes
            figsize (tuple, optional): Figure size
        """
        if route_names is None:
            route_names = list(self.routes.keys())
            
        plt.figure(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(route_names)))
        
        for i, route_name in enumerate(route_names):
            if route_name not in self.routes:
                print(f"Route {route_name} not found")
                continue
                
            df = self.routes[route_name]
            
            # Create a rolling mean to smooth the gradient profile
            gradient_smooth = df['gradient'].rolling(window=10, center=True).mean()
            
            plt.plot(df['cumulative_distance'], gradient_smooth, 
                    label=f"{route_name} (smoothed)", color=colors[i])
            
        plt.xlabel('Distance (km)')
        plt.ylabel('Gradient (%)')
        plt.title('Gradient Profiles (10-point Rolling Average)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_terrain_distribution(self, route_names=None, figsize=(12, 6)):
        """
        Plot terrain distribution for the specified routes
        
        Args:
            route_names (list, optional): Names of routes to plot. If None, plots all routes
            figsize (tuple, optional): Figure size
        """
        if not self.google_maps_api_key:
            print("No Google Maps API key provided, terrain data unavailable")
            return
            
        if route_names is None:
            route_names = list(self.routes.keys())
            
        terrain_data = {}
        
        for route_name in route_names:
            if route_name not in self.route_metrics or 'terrain_distribution' not in self.route_metrics[route_name]:
                print(f"No terrain data available for route {route_name}")
                continue
                
            terrain_data[route_name] = self.route_metrics[route_name]['terrain_distribution']
            
        if not terrain_data:
            print("No terrain data available for any routes")
            return
            
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(terrain_data).T * 100  # Convert to percentage
        
        plt.figure(figsize=figsize)
        df.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())
        plt.xlabel('Route')
        plt.ylabel('Percentage (%)')
        plt.title('Terrain Distribution')
        plt.legend(title='Terrain Type')
        plt.tight_layout()
        plt.show()
    
    def plot_difficulty_radar(self, route_names=None, figsize=(10, 8)):
        """
        Plot a radar chart comparing route difficulties
        
        Args:
            route_names (list, optional): Names of routes to plot. If None, plots all routes
            figsize (tuple, optional): Figure size
        """
        if route_names is None:
            route_names = list(self.routes.keys())
            
        # Filter for routes that exist
        route_names = [name for name in route_names if name in self.route_metrics]
        
        if not route_names:
            print("No routes to compare")
            return
            
        # Categories for the radar chart
        categories = ['Distance', 'Elevation Gain', 'Max Gradient', 
                     'Terrain Variability', 'Steep Sections']
        
        # Prepare the data
        values = []
        for route_name in route_names:
            metrics = self.route_metrics[route_name]
            
            # Normalize each metric to 0-100 scale
            distance_norm = min(metrics['total_distance_km'] / 450 * 100, 100)
            elev_gain_norm = min(metrics['elevation_gain_m'] / 5000 * 100, 100)
            max_gradient_norm = min(metrics['max_gradient_percent'] / 20 * 100, 100)
            terrain_var_norm = min(metrics['terrain_variability_index'] / 5 * 100, 100)
            steep_norm = min(metrics['steep_sections_percent'] / 20 * 100, 100)
            
            values.append([
                distance_norm,
                elev_gain_norm,
                max_gradient_norm,
                terrain_var_norm,
                steep_norm
            ])
        
        # Create the radar chart
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        
        # Number of categories
        N = len(categories)
        
        # Compute angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Plot each route
        colors = plt.cm.tab10(np.linspace(0, 1, len(route_names)))
        
        for i, route_name in enumerate(route_names):
            values_route = values[i]
            values_route += values_route[:1]  # Close the loop
            
            ax.plot(angles, values_route, linewidth=2, linestyle='solid', 
                   label=route_name, color=colors[i])
            ax.fill(angles, values_route, alpha=0.1, color=colors[i])
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Draw y-labels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], 
                  color="grey", size=8)
        plt.ylim(0, 100)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Route Difficulty Comparison', size=15, y=1.1)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_map(self, route_names=None, color_by='gradient'):
        """
        Create an interactive map showing the routes with color-coded paths
        
        Args:
            route_names (list, optional): Names of routes to display. If None, displays all routes
            color_by (str): Attribute to color the path by ('gradient', 'elevation', 'terrain')
            
        Returns:
            folium.Map: Interactive map object
        """
        if route_names is None:
            route_names = list(self.routes.keys())
            
        # Find center point for map
        all_lats = []
        all_lons = []
        
        for route_name in route_names:
            if route_name in self.routes:
                all_lats.extend(self.routes[route_name]['latitude'])
                all_lons.extend(self.routes[route_name]['longitude'])
        
        if not all_lats or not all_lons:
            print("No route data available")
            return None
            
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
        # Create the base map
        route_map = folium.Map(location=[center_lat, center_lon], zoom_start=8)
        
        # Color mapping function
        def get_color(value, min_val, max_val, cmap=plt.cm.viridis):
            norm = Normalize(vmin=min_val, vmax=max_val)
            rgb = cmap(norm(value))[:3]  # Take only RGB, not alpha
            return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        # Add each route
        for i, route_name in enumerate(route_names):
            if route_name not in self.routes:
                continue
                
            df = self.routes[route_name]
            
            # Determine coloring
            if color_by == 'gradient':
                # Use a rolling mean for smoother colors
                values = df['gradient'].rolling(window=10, center=True).mean().fillna(0)
                color_map = plt.cm.RdYlGn_r  # Red for steep, green for flat
                min_val, max_val = -5, 15  # Gradient range
                popup_text = 'Gradient: {:.1f}%'
            elif color_by == 'elevation':
                values = df['elevation']
                color_map = plt.cm.terrain
                min_val, max_val = df['elevation'].min(), df['elevation'].max()
                popup_text = 'Elevation: {:.0f}m'
            elif color_by == 'terrain' and route_name in self.terrain_data:
                # Skip this for now as terrain is categorical
                continue
            else:
                # Default: use a fixed color per route
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
                line_color = colors[i % len(colors)]
                
                # Create a simple polyline
                points = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
                folium.PolyLine(
                    points, 
                    color=line_color,
                    weight=4,
                    opacity=0.8,
                    popup=route_name
                ).add_to(route_map)
                continue
            
            # Create segments with color based on the selected attribute
            for j in range(len(df) - 1):
                if j % 5 != 0:  # Skip some points for better performance
                    continue
                    
                point1 = [df.iloc[j]['latitude'], df.iloc[j]['longitude']]
                point2 = [df.iloc[j+1]['latitude'], df.iloc[j+1]['longitude']]
                
                # Get color
                value = values.iloc[j]
                color = get_color(value, min_val, max_val, cmap=color_map)
                
                # Create popup text
                popup = f"{route_name}<br>{popup_text.format(value)}"
                
                folium.PolyLine(
                    [point1, point2],
                    color=color,
                    weight=4,
                    opacity=0.8,
                    popup=popup
                ).add_to(route_map)
            
            # Add markers for each segment start
            if route_name in self.segment_summaries:
                for _, segment in self.segment_summaries[route_name].iterrows():
                    start_km = segment['start_km']
                    
                    # Find the closest point to this distance
                    closest_idx = (df['cumulative_distance'] - start_km).abs().idxmin()
                    
                    marker_lat = df.iloc[closest_idx]['latitude']
                    marker_lon = df.iloc[closest_idx]['longitude']
                    
                    # Create popup content
                    popup_content = f"""
                    <b>{route_name} - Segment {segment['segment']}</b><br>
                    Distance: {segment['distance_km']:.1f} km<br>
                    Elevation Gain: {segment['elevation_gain_m']:.0f} m<br>
                    Elevation Loss: {segment['elevation_loss_m']:.0f} m<br>
                    Max Gradient: {segment['max_gradient_percent']:.1f}%<br>
                    Steep Sections: {segment['steep_sections']}
                    """
                    
                    folium.Marker(
                        [marker_lat, marker_lon],
                        popup=folium.Popup(popup_content, max_width=300),
                        icon=folium.Icon(icon='info-sign', color='blue')
                    ).add_to(route_map)
        
        # Add legend
        if color_by in ['gradient', 'elevation']:
            # Create a colormap legend
            from branca.colormap import LinearColormap  # Import from correct location
            colormap = LinearColormap(
                colors=[get_color(v, min_val, max_val, color_map) for v in np.linspace(min_val, max_val, 10)],
                vmin=min_val,
                vmax=max_val,
                caption=f'{color_by.capitalize()} Legend'
            )
            colormap.add_to(route_map)
        
        return route_map
    
    def generate_segment_reports(self, route_name):
        """
        Generate detailed reports for each 125km segment of a route
        
        Args:
            route_name (str): Name of the route
            
        Returns:
            list: List of segment reports
        """
        if route_name not in self.segment_summaries:
            print(f"No segment data available for route {route_name}")
            return []
            
        segments = self.segment_summaries[route_name]
        reports = []
        
        for _, segment in segments.iterrows():
            segment_num = segment['segment']
            start_km = segment['start_km']
            end_km = segment['end_km']
            
            # Basic metrics
            distance_km = segment['distance_km']
            elevation_gain = segment['elevation_gain_m']
            elevation_loss = segment['elevation_loss_m']
            max_elevation = segment['max_elevation_m']
            min_elevation = segment['min_elevation_m']
            max_gradient = segment['max_gradient_percent']
            avg_gradient = segment['avg_gradient_percent']
            steep_sections = segment['steep_sections']
            
            # Calculate difficulty score for this segment
            # (simplified version of the route difficulty calculation)
            distance_score = min(distance_km / self.segment_length * 25, 25)
            elevation_score = min(elevation_gain / 2000 * 35, 35)
            gradient_score = min(max_gradient / 20 * 25, 25)
            steep_score = min(steep_sections / 50 * 15, 15)
            
            difficulty_score = distance_score + elevation_score + gradient_score + steep_score
            
            # Determine difficulty level
            if difficulty_score >= 75:
                difficulty_level = "Very Hard"
            elif difficulty_score >= 50:
                difficulty_level = "Hard"
            elif difficulty_score >= 25:
                difficulty_level = "Moderate"
            else:
                difficulty_level = "Easy"
            
            # Create report
            report = {
                'segment_number': segment_num,
                'start_km': start_km,
                'end_km': end_km,
                'distance_km': distance_km,
                'elevation_gain_m': elevation_gain,
                'elevation_loss_m': elevation_loss,
                'max_elevation_m': max_elevation,
                'min_elevation_m': min_elevation,
                'elevation_range_m': max_elevation - min_elevation,
                'max_gradient_percent': max_gradient,
                'avg_gradient_percent': avg_gradient,
                'steep_sections': steep_sections,
                'difficulty_score': difficulty_score,
                'difficulty_level': difficulty_level,
                'key_challenges': []
            }
            
            # Add key challenges based on metrics
            if elevation_gain > 1000:
                report['key_challenges'].append(f"Significant climbing ({elevation_gain:.0f}m)")
            
            if max_gradient > 10:
                report['key_challenges'].append(f"Steep sections (max {max_gradient:.1f}%)")
                
            if steep_sections > 20:
                report['key_challenges'].append(f"Many steep sections ({steep_sections} sections >8%)")
                
            if report['elevation_range_m'] > 1500:
                report['key_challenges'].append(f"Large elevation range ({report['elevation_range_m']:.0f}m)")
                
            # Add terrain challenges if available
            if route_name in self.terrain_data:
                # Get terrain data for this segment
                terrain_df = self.terrain_data[route_name]
                segment_terrain = terrain_df[(terrain_df['distance'] >= start_km) & 
                                           (terrain_df['distance'] <= end_km)]
                
                if not segment_terrain.empty:
                    terrain_counts = segment_terrain['terrain_type'].value_counts()
                    most_common = terrain_counts.idxmax()
                    
                    if most_common == 'mountainous':
                        report['key_challenges'].append("Mountainous terrain")
                    elif most_common == 'hilly':
                        report['key_challenges'].append("Hilly terrain")
                        
            # If no challenges identified, add a default message
            if not report['key_challenges']:
                report['key_challenges'].append("No significant challenges identified")
                
            reports.append(report)
            
        return reports
    
    def print_segment_summaries(self, route_name):
        """
        Print readable summaries for each 125km segment of a route
        
        Args:
            route_name (str): Name of the route
        """
        reports = self.generate_segment_reports(route_name)
        
        if not reports:
            return
            
        print(f"\n{'=' * 80}")
        print(f"ROUTE SEGMENT SUMMARIES FOR: {route_name}")
        print(f"{'=' * 80}")
        
        for report in reports:
            print(f"\nSEGMENT {report['segment_number']}: {report['start_km']:.1f}km to {report['end_km']:.1f}km "
                 f"({report['distance_km']:.1f}km)")
            print(f"{'·' * 80}")
            print(f"Difficulty: {report['difficulty_level']} ({report['difficulty_score']:.1f}/100)")
            print(f"Elevation: Gain {report['elevation_gain_m']:.0f}m | Loss {report['elevation_loss_m']:.0f}m | "
                 f"Range {report['elevation_range_m']:.0f}m")
            print(f"Gradient: Avg {report['avg_gradient_percent']:.1f}% | Max {report['max_gradient_percent']:.1f}% | "
                 f"Steep sections: {report['steep_sections']}")
            
            print("Key Challenges:")
            for challenge in report['key_challenges']:
                print(f"  • {challenge}")
                
        print(f"\n{'=' * 80}\n")
    
    def compare_routes_table(self, route_names=None):
        """
        Generate a comparison table of routes with their key metrics
        
        Args:
            route_names (list, optional): Names of routes to compare. If None, compares all routes
        """
        if route_names is None:
            route_names = list(self.routes.keys())
            
        metrics = self.get_route_metrics()
        
        if metrics.empty:
            print("No route metrics available")
            return
            
        # Filter for requested routes
        metrics = metrics.loc[metrics.index.isin(route_names)]
        
        if metrics.empty:
            print("No metrics found for the requested routes")
            return
            
        # Select key metrics for display
        display_metrics = metrics[[
            'total_distance_km', 
            'elevation_gain_m',
            'elevation_loss_m',
            'max_elevation_m',
            'avg_gradient_percent',
            'max_gradient_percent',
            'steep_sections_count',
            'difficulty_score'
        ]].copy()
        
        # Rename columns for better display
        display_metrics.columns = [
            'Distance (km)',
            'Elevation Gain (m)',
            'Elevation Loss (m)',
            'Max Elevation (m)',
            'Avg Gradient (%)',
            'Max Gradient (%)',
            'Steep Sections',
            'Difficulty Score'
        ]
        
        # Format numbers
        pd.options.display.float_format = '{:.1f}'.format
        
        # Rank routes by difficulty
        difficulty_rank = metrics['difficulty_score'].rank(ascending=False).astype(int)
        display_metrics.insert(0, 'Difficulty Rank', difficulty_rank)
        
        # Sort by difficulty rank
        display_metrics = display_metrics.sort_values('Difficulty Rank')
        
        print("\nROUTE COMPARISON TABLE")
        print("=" * 100)
        print(display_metrics)
        print("=" * 100)
        
    def export_to_html(self, output_file, route_names=None):
        """
        Export analysis results to an HTML report
        
        Args:
            output_file (str): Path to save the HTML report
            route_names (list, optional): Names of routes to include. If None, includes all routes
        """
        if route_names is None:
            route_names = list(self.routes.keys())
            
        # Start HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GPX Route Analysis: LA to Vegas</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .segment {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; }}
                .easy {{ background-color: #d4edda; }}
                .moderate {{ background-color: #fff3cd; }}
                .hard {{ background-color: #ffe5d0; }}
                .very-hard {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>LA to Vegas Route Analysis</h1>
            <p>Analysis Date: {time.strftime('%Y-%m-%d')}</p>
        """
        
        # Add route comparison table
        metrics = self.get_route_metrics()
        if not metrics.empty:
            html_content += """
            <h2>Route Comparison</h2>
            <table>
                <tr>
                    <th>Route</th>
                    <th>Difficulty Score</th>
                    <th>Distance (km)</th>
                    <th>Elevation Gain (m)</th>
                    <th>Max Gradient (%)</th>
                    <th>Steep Sections</th>
                </tr>
            """
            
            # Sort routes by difficulty
            metrics = metrics.sort_values('difficulty_score', ascending=False)
            
            for route_name in metrics.index:
                if route_name not in route_names:
                    continue
                    
                route_metrics = metrics.loc[route_name]
                html_content += f"""
                <tr>
                    <td>{route_name}</td>
                    <td>{route_metrics['difficulty_score']:.1f}</td>
                    <td>{route_metrics['total_distance_km']:.1f}</td>
                    <td>{route_metrics['elevation_gain_m']:.0f}</td>
                    <td>{route_metrics['max_gradient_percent']:.1f}</td>
                    <td>{route_metrics['steep_sections_count']}</td>
                </tr>
                """
                
            html_content += "</table>"
            
        # Add segment summaries for each route
        for route_name in route_names:
            if route_name not in self.segment_summaries:
                continue
                
            html_content += f"<h2>Segment Analysis: {route_name}</h2>"
            
            reports = self.generate_segment_reports(route_name)
            
            for report in reports:
                difficulty = report['difficulty_level'].lower().replace(' ', '-')
                
                html_content += f"""
                <div class="segment {difficulty}">
                    <h3>Segment {report['segment_number']}: {report['start_km']:.1f}km to {report['end_km']:.1f}km ({report['distance_km']:.1f}km)</h3>
                    <p><strong>Difficulty:</strong> {report['difficulty_level']} ({report['difficulty_score']:.1f}/100)</p>
                    
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Elevation Gain</td>
                            <td>{report['elevation_gain_m']:.0f}m</td>
                        </tr>
                        <tr>
                            <td>Elevation Loss</td>
                            <td>{report['elevation_loss_m']:.0f}m</td>
                        </tr>
                        <tr>
                            <td>Max Elevation</td>
                            <td>{report['max_elevation_m']:.0f}m</td>
                        </tr>
                        <tr>
                            <td>Elevation Range</td>
                            <td>{report['elevation_range_m']:.0f}m</td>
                        </tr>
                        <tr>
                            <td>Average Gradient</td>
                            <td>{report['avg_gradient_percent']:.1f}%</td>
                        </tr>
                        <tr>
                            <td>Maximum Gradient</td>
                            <td>{report['max_gradient_percent']:.1f}%</td>
                        </tr>
                        <tr>
                            <td>Steep Sections</td>
                            <td>{report['steep_sections']}</td>
                        </tr>
                    </table>
                    
                    <h4>Key Challenges:</h4>
                    <ul>
                """
                
                for challenge in report['key_challenges']:
                    html_content += f"<li>{challenge}</li>"
                    
                html_content += """
                    </ul>
                </div>
                """
                
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        print(f"Report exported to {output_file}")


# Example usage
if __name__ == "__main__":
    # Create analyzer with Google Maps API key
    analyzer = GPXRouteAnalyzer(segment_length=125, google_maps_api_key="YOUR_GOOGLE_MAPS_API_KEY")
    
    # Load GPX files
    analyzer.load_gpx_file("route1.gpx", "Northern Route")
    analyzer.load_gpx_file("route2.gpx", "Mountain Route")
    analyzer.load_gpx_file("route3.gpx", "Desert Route")
    
    # Compare routes
    analyzer.compare_routes_table()
    
    # Print detailed segment summaries
    analyzer.print_segment_summaries("Northern Route")
    
    # Create visualizations
    analyzer.plot_elevation_profiles()
    analyzer.plot_gradient_profiles()
    analyzer.plot_difficulty_radar()
    
    # Create interactive map
    route_map = analyzer.create_interactive_map(color_by='gradient')
    
    # Export to HTML report
    analyzer.export_to_html("la_to_vegas_route_analysis.html")