"""
Plot helper functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def set_equal_axis(ax, xs, ys, zs, scale=1.0, dim3=True):
    """Helper function to set equal axis
    
    Args:
        ax (Axes3DSubplot): matplotlib 3D axis, created by `ax = fig.add_subplot(projection='3d')`
        xlims (list): 2-element list containing min and max value of x
        ylims (list): 2-element list containing min and max value of y
        zlims (list): 2-element list containing min and max value of z
        scale (float): scaling factor along x,y,z
        dim3 (bool): whether to also set z-limits (True for 3D plots)
    """
    # get limits of each axis
    xlims = [min(xs), max(xs)]
    ylims = [min(ys), max(ys)]
    zlims = [min(zs), max(zs)]
    # compute max required range
    max_range = np.array([max(xlims)-min(xlims), max(ylims)-min(ylims), max(zlims)-min(zlims)]).max() / 2.0
    # compute mid-point along each axis
    mid_x = (max(xlims) + min(xlims)) * 0.5
    mid_y = (max(ylims) + min(ylims)) * 0.5
    mid_z = (max(zlims) + min(zlims)) * 0.5
    # set limits to axis
    if dim3==True:
        ax.set_box_aspect((max_range, max_range, max_range))
    ax.set_xlim(mid_x - max_range*scale, mid_x + max_range*scale)
    ax.set_ylim(mid_y - max_range*scale, mid_y + max_range*scale)
    if dim3==True:
        ax.set_zlim(mid_z - max_range*scale, mid_z + max_range*scale)
    return