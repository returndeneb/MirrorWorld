import os
import argparse
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import pyvista as pv
import osmnx as ox
import rioxarray
import cmocean
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator, griddata
from typing import Dict, Any, Tuple

# --- Data Utility Functions ---

def get_building_height(building_data: pd.Series) -> float:
    height = building_data.get('height')
    levels = building_data.get('building:levels')
    if height:
        try:
            return float(str(height).split(';')[0])
        except (ValueError, TypeError): pass
    if levels:
        try:
            return int(levels) * 3.5
        except (ValueError, TypeError): pass
    return np.random.uniform(10, 40)

def load_textures() -> Dict[str, pv.Texture] | None:
    texture_dir = "textures"
    texture_files = {
        'grass': os.path.join(texture_dir, 'grass.jpg'),
        'brick': os.path.join(texture_dir, 'brick.jpg'),
        'roof': os.path.join(texture_dir, 'roof.jpg'),
        'asphalt': os.path.join(texture_dir, 'asphalt.jpg'),
        'custom_facade': os.path.join(texture_dir, 'building_facade.png'), # Custom texture
    }
    os.makedirs(texture_dir, exist_ok=True)
    
    loaded_textures = {}
    print("Loading textures...")
    for name, path in texture_files.items():
        if os.path.exists(path):
            try:
                loaded_textures[name] = pv.read_texture(path)
            except Exception as e:
                print(f"Warning: Could not load texture '{name}' from {path}. Error: {e}")
    
    if not loaded_textures:
        print("Warning: No textures found. Rendering with default colors.")
        return None
    return loaded_textures

def load_or_fetch_data(place_name: str, buffer_dist: int, api_key: str) -> Dict[str, Any]:
    place_slug = place_name.split(',')[0].replace(' ', '_').lower()
    data_dir = f"data_{place_slug}"
    os.makedirs(data_dir, exist_ok=True)
    paths = {
        'bldgs': os.path.join(data_dir, "buildings.gpkg"),
        'roads': os.path.join(data_dir, "roads.gpkg"),
        'nodes': os.path.join(data_dir, "nodes.gpkg"),
        'dem': os.path.join(data_dir, "dem.tif")
    }

    if all(os.path.exists(p) for p in paths.values()):
        print(f"Loading cached data for '{place_name}'.")
        gdf_bldgs = gpd.read_file(paths['bldgs'])
        gdf_roads = gpd.read_file(paths['roads'])
        gdf_nodes = gpd.read_file(paths['nodes']).set_index('osmid')
        utm_crs = gdf_bldgs.estimate_utm_crs()
    else:
        print(f"Fetching online data for '{place_name}'.")
        gdf_place = ox.geocode_to_gdf(place_name)
        utm_crs = gdf_place.estimate_utm_crs()
        gdf_place_utm = gdf_place.to_crs(utm_crs)
        gdf_place_utm.geometry = gdf_place_utm.geometry.buffer(buffer_dist)
        gdf_place_buffered = gdf_place_utm.to_crs(gdf_place.crs)
        master_polygon = gdf_place_buffered['geometry'].iloc[0]
        
        gdf_bldgs = ox.features_from_polygon(master_polygon, {"building": True})
        gdf_bldgs.to_file(paths['bldgs'], driver="GPKG")

        graph = ox.graph_from_polygon(master_polygon, network_type='all')
        gdf_nodes, gdf_roads = ox.graph_to_gdfs(graph)
        gdf_roads.to_file(paths['roads'], driver="GPKG")
        gdf_nodes.reset_index().to_file(paths['nodes'], driver="GPKG")

        bounds = pd.concat([gdf_bldgs['geometry'], gdf_roads['geometry']], ignore_index=True).total_bounds
        url = "https://portal.opentopography.org/API/globaldem"
        params = {'demtype': 'SRTMGL1', 'south': bounds[1], 'north': bounds[3], 'west': bounds[0], 'east': bounds[2], 'outputFormat': 'GTiff', 'API_Key': api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        with open(paths['dem'], 'wb') as f:
            f.write(response.content)

    return {'bldgs': gdf_bldgs, 'roads': gdf_roads, 'nodes': gdf_nodes, 'dem_path': paths['dem'], 'utm_crs': utm_crs}

# --- 3D Generation and Rendering ---

def prepare_terrain(dem_path: str, utm_crs: Any, vertical_exaggeration: float) -> Tuple[pv.StructuredGrid, RegularGridInterpolator]:
    dem_utm = rioxarray.open_rasterio(dem_path, masked=True).rio.reproject(utm_crs, resampling=Resampling.cubic)
    elevation = dem_utm.values.squeeze()
    elevation[elevation < -1000] = np.nan
    if np.isnan(elevation).any():
        x_grid, y_grid = np.meshgrid(np.arange(elevation.shape[1]), np.arange(elevation.shape[0]))
        valid_points = ~np.isnan(elevation)
        elevation = griddata((x_grid[valid_points], y_grid[valid_points]), elevation[valid_points], (x_grid, y_grid), method='cubic', fill_value=np.nanmin(elevation))

    elevation_scaled = elevation * vertical_exaggeration
    x_coords, y_coords = dem_utm.x.values, dem_utm.y.values
    
    if y_coords[0] > y_coords[-1]:
        y_coords = y_coords[::-1]
        elevation_scaled = np.flip(elevation_scaled, axis=0)
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    terrain_mesh = pv.StructuredGrid(xx, yy, elevation_scaled)
    terrain_mesh['elevation'] = elevation_scaled.ravel(order='F')
    
    z_interpolator = RegularGridInterpolator((y_coords, x_coords), elevation_scaled, bounds_error=False, fill_value=np.nanmin(elevation_scaled))
    return terrain_mesh, z_interpolator

def populate_plotter(plotter: pv.Plotter, data: Dict[str, Any], z_interpolator: RegularGridInterpolator, translation_vector: Tuple[float, float, float], textures: Dict[str, pv.Texture] | None, target_building_id: int):
    gdf_bldgs_proj = data['bldgs'].to_crs(data['utm_crs']).reset_index() # Use osmid as a column
    gdf_roads_proj = data['roads'].to_crs(data['utm_crs'])
    gdf_nodes_proj = data['nodes'].to_crs(data['utm_crs'])
    nan_z_fallback = z_interpolator.fill_value

    # Create building meshes
    for _, bldg in gdf_bldgs_proj.iterrows():
        try:
            if bldg.geometry.is_empty or not bldg.geometry.is_valid: continue
            
            points_2d = np.array(bldg.geometry.exterior.coords)
            centroid = bldg.geometry.centroid
            ground_z = z_interpolator((centroid.y, centroid.x)).item() + 0.1
            if np.isnan(ground_z): ground_z = nan_z_fallback
            
            base_points = np.pad(points_2d, ((0, 0), (0, 1)), 'constant', constant_values=ground_z)[:-1]
            if len(base_points) < 3: continue
            
            floor = pv.PolyData(base_points, faces=[len(base_points)] + list(range(len(base_points))))
            height = get_building_height(bldg)
            
            if textures:
                # Select texture based on building ID
                if bldg['osmid'] == target_building_id and 'custom_facade' in textures:
                    wall_texture = textures.get('custom_facade')
                else:
                    wall_texture = textures.get('brick')
                
                roof_texture = textures.get('roof')

                walls_mesh = floor.extrude((0, 0, height), capping=False)
                walls_mesh.texture_map_to_plane(inplace=True)
                roof_mesh = floor.copy().translate((0, 0, height), inplace=True)
                roof_mesh.texture_map_to_plane(inplace=True)

                walls_mesh.translate(translation_vector, inplace=True)
                roof_mesh.translate(translation_vector, inplace=True)

                plotter.add_mesh(walls_mesh, texture=wall_texture, show_edges=True, edge_color='gray')
                plotter.add_mesh(roof_mesh, texture=roof_texture, show_edges=True, edge_color='gray')
            else:
                building_mesh = floor.extrude((0, 0, height), capping=True).translate(translation_vector, inplace=True)
                plotter.add_mesh(building_mesh, color='ivory', show_edges=True, edge_color='gray')
        except Exception:
            continue
    
    # Create road meshes
    # ... (road logic remains the same) ...

def capture_specific_view(plotter: pv.Plotter, view_latlon: Tuple[float, float], utm_crs: Any, z_interpolator: RegularGridInterpolator, translation_vector: Tuple[float, float, float], heading: float, pitch: float, fov: float, output_filename: str):
    print(f"Capturing specific view to '{output_filename}'...")
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([view_latlon[1]], [view_latlon[0]]), crs="EPSG:4326")
    gdf_utm = gdf.to_crs(utm_crs)
    view_point_utm = gdf_utm.geometry.iloc[0]

    cam_z = z_interpolator((view_point_utm.y, view_point_utm.x)).item() + 1.7
    camera_pos = (view_point_utm.x + translation_vector[0], view_point_utm.y + translation_vector[1], cam_z + translation_vector[2])

    yaw_rad = np.deg2rad(90 - heading)
    pitch_rad = np.deg2rad(pitch)
    dist = 100
    dx = dist * np.cos(pitch_rad) * np.cos(yaw_rad)
    dy = dist * np.cos(pitch_rad) * np.sin(yaw_rad)
    dz = dist * np.sin(pitch_rad)
    focal_point = (camera_pos[0] + dx, camera_pos[1] + dy, camera_pos[2] + dz)

    plotter.camera.position = camera_pos
    plotter.camera.focal_point = focal_point
    plotter.camera.viewup = (0, 0, 1)
    plotter.camera.view_angle = fov
    plotter.screenshot(output_filename, transparent_background=True)
    print(f"View captured successfully: '{output_filename}'")

# --- Main Execution ---

def main(place_name: str, api_key: str, buffer_dist: int, vertical_exaggeration: float):
    # ... (Data loading and terrain preparation as before) ...
    data = load_or_fetch_data(place_name, buffer_dist, api_key)
    terrain_mesh, z_interpolator = prepare_terrain(data['dem_path'], data['utm_crs'], vertical_exaggeration)
    
    bldgs_proj = data['bldgs'].to_crs(data['utm_crs'])
    bounds = bldgs_proj.total_bounds
    center_x, center_y = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    center_z = np.nanmean(terrain_mesh.points[:, 2])
    translation_vector = (-center_x, -center_y, -center_z)

    textures = load_textures()
    terrain_surface = terrain_mesh.extract_surface().translate(translation_vector, inplace=True)

    # Define the building to receive the custom texture
    # This ID would come from exploring the data for your target building
    TARGET_BUILDING_OSMID = 36985474 # Example OSM ID for N Seoul Tower

    # Off-screen plotter for screenshot
    plotter_ss = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter_ss.set_background('lightblue')
    if textures and 'grass' in textures:
        terrain_surface.texture_map_to_plane(inplace=True, use_bounds=True)
        plotter_ss.add_mesh(terrain_surface, texture=textures.get('grass'))
    else:
        plotter_ss.add_mesh(terrain_surface, cmap=cmocean.cm.topo)
    populate_plotter(plotter_ss, data, z_interpolator, translation_vector, textures, TARGET_BUILDING_OSMID)
    capture_specific_view(plotter_ss, (37.55032, 126.98781), data['utm_crs'], z_interpolator, translation_vector, heading=63.8, pitch=-9.6, fov=75.0, output_filename="namsan_tower_view.png")
    plotter_ss.close()

    # Interactive plotter for display
    plotter_interactive = pv.Plotter(window_size=[1200, 800], lighting='light_kit')
    plotter_interactive.set_background('lightblue')
    if textures and 'grass' in textures:
        plotter_interactive.add_mesh(terrain_surface, texture=textures.get('grass'))
    else:
        plotter_interactive.add_mesh(terrain_surface, cmap=cmocean.cm.topo, scalar_bar_args={'title': 'Elevation (m)'})
    populate_plotter(plotter_interactive, data, z_interpolator, translation_vector, textures, TARGET_BUILDING_OSMID)
    plotter_interactive.show_grid(color='gray')
    print("Rendering complete. Close the interactive window to exit.")
    plotter_interactive.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a 3D terrain model from a specified place name.")
    parser.add_argument("--place", type=str, default="Namsan Tower, Seoul", help="Name of the location to model.")
    parser.add_argument("--buffer", type=int, default=800, help="Buffer radius in meters to fetch data for.")
    parser.add_argument("--z_scale", type=float, default=1.5, help="Vertical exaggeration factor for terrain.")
    parser.add_argument("--api_key", type=str, required=True, help="Your OpenTopography API key.")
    args = parser.parse_args()
    main(args.place, args.api_key, args.buffer, args.z_scale)