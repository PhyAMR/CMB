"""
This module contains functions for manipulating and analyzing CMB maps,
typically provided as HEALPix arrays.
"""

import numpy as np
import healpy as hp

def map_rot_refl(map_data):
    """
    Performs a rotation and reflection of a HEALPix map to obtain the opposite view.
    This is equivalent to observing the sky from the opposite direction.

    Args:
        map_data (np.ndarray): The input HEALPix map.

    Returns:
        np.ndarray: The transformed map representing the opposite view.
    """
    # Create an empty map with the same resolution as the input map.
    opposite_map = np.zeros_like(map_data)
    
    # Get the NSIDE parameter of the map, which defines its resolution.
    nside = hp.get_nside(map_data)
    
    # Get the longitude (l) and latitude (b) for each pixel in the map.
    l, b = hp.pix2ang(nside, np.arange(len(map_data)), lonlat=True)

    # To get the opposite view, invert the latitude and add 180 degrees to the longitude.
    b_opposite = -b
    l_opposite = l + 180
    
    # Find the pixel indices corresponding to the new (opposite) coordinates.
    opposite_index = hp.ang2pix(nside, l_opposite, b_opposite, lonlat=True)
    
    # Populate the opposite map with the data from the original map at the new indices.
    opposite_map = map_data[opposite_index]
    
    return opposite_map

def estimate_coef(x, y):
    """
    Estimates the coefficients of a simple linear regression model (y = b_0 + b_1*x).

    Args:
        x (np.ndarray): The independent variable data.
        y (np.ndarray): The dependent variable data.

    Returns:
        tuple: A tuple containing the estimated coefficients (b_0, b_1).
            - b_0: The y-intercept of the regression line.
            - b_1: The slope of the regression line.
    """
    # Number of observations/points.
    n = np.size(x)
  
    # Mean of the x and y vectors.
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # Calculate the cross-deviation and the deviation about x.
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
  
    # Calculate the regression coefficients.
    b_1 = SS_xy / SS_xx  # Slope
    b_0 = m_y - b_1 * m_x  # Intercept
  
    return (b_0, b_1)