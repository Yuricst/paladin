"""
Newton Raphson iteration
"""

import numpy as np
import numpy.linalg as la
from copy import copy
from numba import njit


@njit
def _newtonraphson_iteration(xi, ferr, df, alpha=1.0):
	"""Apply Newton-Raphson iteration
	Args:
		xi (np.array): length-n array of free variables
		ferr (np.array): length n array of residuals
		df (np.array): n by n np.array of Jacobian
		alpha (real): update multiplier, (0,1.0]
		
	Returns:
		(np.array): length-n array of new free variables
	"""
	assert 0 < alpha <= 1.0, "alpha on Newton raphson must be in (0,1]"
	xi_vert = np.reshape(xi, (len(xi), -1))
	ferr_vert = np.reshape(ferr, (len(ferr), -1))
	xii_vert = xi_vert - alpha * np.dot(la.pinv(df), ferr_vert)
	return np.reshape(xii_vert, (len(xii_vert),))


@njit
def _leastsquare_iteration(xi, ferr, df):
	"""Apply least-square iteration
	
	Args:
		xi (np.array): length-n array of free variables
		ferr (np.array): length m array of residuals
		df (np.array): m by n np.array of Jacobian
		
	Returns:
		(np.array): length-n array of new free variables
	"""
	xi_vert = np.reshape(xi, (len(xi), -1))
	ferr_vert = np.reshape(ferr, (len(ferr), -1))
	mapDF = np.dot( la.pinv( np.dot(np.transpose(df), df) ) , np.transpose(df) )
	xii_vert = xi_vert - np.dot( mapDF, ferr_vert)
	return np.reshape(xii_vert, (len(xii_vert),))


@njit
def _minimumnorm_iteration(xi, ferr, df, damping=1.0):
	"""Apply minimum-norm iteration
	
	Args:
		xi (np.array): length-n array of free variables
		ferr (np.array): length m array of residuals
		df (np.array): m by n np.array of Jacobian
		
	Returns:
		(np.array): length-n array of new free variables
	"""
	xi_vert = np.reshape(xi, (len(xi), -1))
	ferr_vert = np.reshape(ferr, (len(ferr), -1))
	mapDF = np.dot( np.transpose(df), la.pinv( np.dot( df,np.transpose(df)) ) )
	xii_vert = xi_vert - damping * np.dot( mapDF, ferr_vert)
	return np.reshape(xii_vert, (len(xii_vert),))

