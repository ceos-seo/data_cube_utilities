import folium
import itertools    
import math
import numpy as np  

def _degree_to_zoom_level(l1, l2, margin = 0.0):
    degree = abs(l1 - l2) * (1 + margin)
    zoom_level_float = math.log(360/degree)/math.log(2)
    zoom_level_int = int(zoom_level_float)
    return zoom_level_int

def display_map(latitude = None, longitude = None, resolution = None):
    """ Returns a centered folium map that outlines lat-lon bounds. This function also adjusts for zoom level based on latitude bounds
    
        Inputs:
            latitude:  a tuple of latitudes bounds in (min,max) format
            longitude: a tuple of longitude bounds in (min,max) format
            resolution: an (optional) tuple in (lat,lon) format used to draw a grid on your map. Gridding starts at top left corner.
        Output:
            follium.Map object 
    """
   
    assert latitude is not None
    assert longitude is not None
   

    ###### ###### ######   CALC ZOOM LEVEL     ###### ###### ######

    margin = -0.5
    zoom_bias = 0
    
    lat_zoom_level = _degree_to_zoom_level(*latitude, margin = margin) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(*longitude, margin = margin) + zoom_bias
    zoom_level = min(lat_zoom_level, lon_zoom_level) 

    ###### ###### ######   CENTER POINT        ###### ###### ######
    
    center = [np.mean(latitude), np.mean(longitude)]

    ###### ###### ######   CREATE MAP         ###### ###### ######
    
    map_hybrid = folium.Map(
        location=center,
        zoom_start=zoom_level, 
        tiles=" http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google"
    )
    
    ###### ###### ######   RESOLUTION GRID    ###### ###### ######
    
    if resolution is not None:
        res_lat, res_lon = resolution

        lats = np.arange(*latitude, abs(res_lat))
        lons = np.arange(*longitude, abs(res_lon))

        vertical_grid   = map(lambda x :([x[0][0],x[1]],[x[0][1],x[1]]),itertools.product([latitude],lons))
        horizontal_grid = map(lambda x :([x[1],x[0][0]],[x[1],x[0][1]]),itertools.product([longitude],lats))

        for segment in vertical_grid:
            folium.features.PolyLine(segment, color = 'white', opacity = 0.3).add_to(map_hybrid)    
        
        for segment in horizontal_grid:
            folium.features.PolyLine(segment, color = 'white', opacity = 0.3).add_to(map_hybrid)   
    
    ###### ###### ######     BOUNDING BOX     ###### ###### ######
    
    line_segments = [(latitude[0],longitude[0]),
                     (latitude[0],longitude[1]),
                     (latitude[1],longitude[1]),
                     (latitude[1],longitude[0]),
                     (latitude[0],longitude[0])
                    ]
    
    
    
    map_hybrid.add_child(
        folium.features.PolyLine(
            locations=line_segments,
            color='red',
            opacity=0.8)
    )

    map_hybrid.add_child(folium.features.LatLngPopup())        

    return map_hybrid


