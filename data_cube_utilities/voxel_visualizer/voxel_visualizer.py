import os
import uuid
import re

import psutil

import numpy as np
import xarray as xr

from jinja2 import Template
from IPython.display import HTML

from utils.data_cube_utilities.dc_time import _n64_to_datetime, dt_to_str

VOXEL_VIS_WEB_SVR_CMD = 'python3 server.py &'

def voxel_visualize(da: xr.DataArray, **kwargs):
    """
    Show a 3D visualization of a boolean xarray `xr`.

    It creates an `iframe` DOM element in the cell's output in Jupyter.

    The camera can be controlled with either:
    1. The mouse and arrow keys OR
    2. Buttons on the right side (hideable)

    There is a slider on the left side with 2 modes - Range and Select.
    * Range: This mode shows layers (time slices) after the selected time
             (shown as text above the slider) at opacity `voxel_opacity`.
             Layers before the selected time are shown in a lower opacity 
             (more translucent).
    * Select: This mode shows only the selected layer at opacity `voxel_opacity`.
              Layers other than the selected time are shown in a lower opacity 
              (more translucent).
    
    The visualization is created with Three.js.

    Parameters
    ----------
    da: xr.DataArray
        The boolean DataArray to show in 3D.
    x_scale, y_scale, z_scale: numeric
        Distance scale factors for voxels the x, y, and z dimensions (default 1).
    distance_scale: numeric
        Distance scale factor for voxels in all dimensions (default 1).
    voxel_size: numeric
        The initial size of the voxels (default 3).
    voxel_opacity: float
        The opacity of the voxels (range: [0,1], default 0.5).
    show_stats: bool
        Whether to show the stats such as FPS (default False).
    show_controls: bool
        Whether to show the controls (default True).
    """
    cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    def _launch_server_if_not_running():
        # Determine if the server is running.
        process_cmds = (p.cmdline() for p in psutil.process_iter())
        cmd_found = False
        for cmd in process_cmds:
            for token in VOXEL_VIS_WEB_SVR_CMD.split():
                if token != '&' and token not in cmd:
                    break
                cmd_found = True
                break
            if cmd_found:
                break
        # If the server is not running, start it.
        if not cmd_found:
            os.system(VOXEL_VIS_WEB_SVR_CMD)
    
    # Ensure the webserver is running.
    _launch_server_if_not_running()
    # Load the voxel visualizer template.
    fs = open('template.html','r')
    template = Template(fs.read())
    fs.close()
    
    if not da.dtype == 'bool':
        raise Exception("You need to pass a boolean xarray.DataArray to use this.")
        
    # Reverse the x dimension.
    da = da.sel(longitude=da.longitude[::-1]).astype(np.int8)

    da_str = str(da.values.tolist())#.replace('array(', '').replace(')', '')#.replace('\n', ',').replace(',,', ',')
    times_str = str([dt_to_str(_n64_to_datetime(time), fmt='%Y-%m-%dT%H:%M:%S.%f') 
                     for time in da.time.values]).replace(',', ',\n')
    # Render the template.
    x_scale = kwargs.get('x_scale', 1)
    assert isinstance(x_scale, (int, float)), "x_scale must be an int or float."
    kwargs['x_scale'] = x_scale
    y_scale = kwargs.get('y_scale', 1)
    assert isinstance(y_scale, (int, float)), "y_scale must be an int or float."
    kwargs['y_scale'] = x_scale
    z_scale = kwargs.get('z_scale', 1)
    assert isinstance(z_scale, (int, float)), "z_scale must be an int or float."
    kwargs['z_scale'] = z_scale
    distance_scale = kwargs.get('distance_scale', 1)
    assert isinstance(distance_scale, (int, float)), "distance_scale must be an int or float."
    kwargs['distance_scale'] = distance_scale
    voxel_size = kwargs.get('voxel_size', 4)
    assert isinstance(voxel_size, (int, float)), "voxel_size must be an int or float."
    kwargs['voxel_size'] = voxel_size
    voxel_opacity = kwargs.get('voxel_opacity', 1)
    assert isinstance(voxel_opacity, (int, float)), "voxel_opacity must be an int or float."
    kwargs['voxel_opacity'] = voxel_opacity
    show_stats = kwargs.setdefault('show_stats', False)
    assert isinstance(show_stats, bool), "show_stats must be a boolean."
    kwargs['show_stats'] = show_stats
    show_controls = kwargs.setdefault('show_controls', True)
    assert isinstance(show_controls, bool), "show_controls must be a boolean."
    kwargs['show_controls'] = show_controls
    filled_template = template.render(data_array=da_str, times=times_str, **kwargs)

    # Remove single line comments and add 
    # line continuation characters (\ in JS).
    # Ensure the HTML is all on one line for the iframe srcdoc.
    filled_template_no_sngl_lne_cmts = []
    for i, line in enumerate(filled_template.splitlines()):
        if re.search('^\s*//', line) is None:
            filled_template_no_sngl_lne_cmts.append(line)
    filled_template_sngl_lne = ''.join(filled_template_no_sngl_lne_cmts)

    # Escape quotes for JS string concatenation.
    filled_template_sngl_lne_esc = filled_template_sngl_lne\
        .replace('\"', '\\"').replace("\'", "\\'")#\

    # "Escape" script tags to avoid closing the script tag
    # containing the substituted filled template HTML string.
    end_scr = '/script>'
    filled_template_sngl_lne_esc_split = \
        re.split(end_scr, filled_template_sngl_lne_esc)
    # Format the strings to form the full string in JS by concat.
    filled_template_sngl_lne_esc_split_fmt = []
    for i, string in enumerate(filled_template_sngl_lne_esc_split):
        # All but first must have end script tag restored.
        # All are enclosed in single quotes.
        if i > 0:
            string = f"\'{end_scr}{string}\'"
        else:
            string = f"\'{string}\'"
        filled_template_sngl_lne_esc_split_fmt.append(string)
    filled_template_sngl_lne_esc_fmt = \
        " + ".join(filled_template_sngl_lne_esc_split_fmt)

    vox_vis_server_port = os.environ['VOXEL_VISUALIZER_PORT']
    iframe = HTML(f"""
    <div id='wrap', style='width:100%; aspect-ratio:2;'>
        <iframe id='voxel_vis_iframe', sandbox='allow-same-origin allow-scripts',
            style='display: block; 
                height:100%; width:99%;' scrolling='no';
                transform-origin: top left>
        </iframe>
    </div>
    <script>
      var hostname = window.location.hostname;
      var static_url = 'http://' + hostname + ':{vox_vis_server_port}/static';
      var srcdoc = {filled_template_sngl_lne_esc_fmt}; """ + """
      srcdoc = srcdoc.replaceAll('static_url', static_url);
      voxel_vis_iframe = document.getElementById('voxel_vis_iframe');
      voxel_vis_iframe.srcdoc = srcdoc;
      
      window.addEventListener("message", onMessage, false);
      function onMessage(event) {
          console.log('message:' + event.data.message);
      } 
    </script>
    """)

    os.chdir(cwd)
    return iframe