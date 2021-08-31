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

def voxel_visualize(da: xr.DataArray):
    """
    Show a 3D visualization of a boolean xarray `xr`.
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
    # Render the template and ensure the 
    # HTML is all on one line for the iframe.
    filled_template = template.render(data_array=da_str, times=times_str)

    # Remove single line comments and add 
    # line continuation characters (\ in JS).
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
    <iframe id='iframe', sandbox='allow-same-origin allow-scripts', width=600, height=350></iframe>
    <script>
      var hostname = window.location.hostname;
      var static_url = 'http://' + hostname + ':{vox_vis_server_port}/static';
      var srcdoc = {filled_template_sngl_lne_esc_fmt}; """ + """
      srcdoc = srcdoc.replaceAll('static_url', static_url);
      document.getElementById('iframe').srcdoc = srcdoc;
    </script>
    """)

    os.chdir(cwd)
    return iframe