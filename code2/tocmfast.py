"""
Some 21cmFast box access stuff for convenient access
"""

import numpy as np
import os
import re
import sys
import scipy.ndimage.filters
import matplotlib.colors as colors

###############################

class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
################################

####################################
# File I/O
###################################

def boxes_list_from_files(basedir, fieldflag = 'tb'):
    """
    Read in 21cmFast datacube boxes and order correctly
    """

    if fieldflag == 'tb':
        prefix = 'delta_T'
    elif fieldflag == 'xh':
        prefix = 'xH_noha'
    elif fieldflag == 'dx':
        prefix = 'updated_smoothed_deltax'
    else:
        raise Exception("fieldflag not defined in boxes_list_From_files")
    
    filenames=os.listdir(basedir)
    box_files=[]
    longdict={}
    for filename in filenames:

        #identify the files of interest by looking for prefix at beginning
        #if filename[0:7] == "delta_T":
        if filename.find(prefix) == 0:
            match=re.search('_z([0-9.]+)',filename)
            z = float(match.group(1))

            longdict[z] = filename
    return longdict

def pk_boxes_list_from_files(basedir, fieldflag = 'tb'):
    """
    Read in 21cmFast power spectrum files and order correctly
    """

    ext = "Output_files/Deldel_T_power_spec/"
    prefix = "ps_no_halos"
    
    filenames=os.listdir(os.path.join(basedir, ext))
    box_files=[]
    longdict={}
    for filename in filenames:

        #identify the files of interest by looking for prefix at beginning
        if filename.find(prefix) == 0:
            match=re.search('_z([0-9.]+)',filename)
            z = float(match.group(1))

            longdict[z] = os.path.join(ext, filename)
    return longdict


def load_binary_data(filename, dtype=np.float32): 
    """ 
    We assume that the data was written 
    with write_binary_data() (little endian). 
    """ 
    f = open(filename, "rb") 
    data = f.read() 
    f.close() 
    _data = np.fromstring(data, dtype) 
    if sys.byteorder == 'big':
        _data = _data.byteswap()
    return _data 


def readtocmbox(filename, DIM=256):
    # read in the data cube located in 21cmFast/Boxes/delta_T*
    data1 = load_binary_data(filename)
    data1.shape = (DIM, DIM, DIM)
    data1 = data1.reshape((DIM, DIM, DIM), order='F')
    return data1

def lookup_xh(basedir, fieldflag = "tb"):
    """
    Get a look up table of the neutral fraction
    """
    xhlookup = {}
    longdict = boxes_list_from_files(basedir, fieldflag)
    for z in sorted(longdict.keys()):
        filename = longdict[z]
        match=re.search('_nf([0-9.]+)',filename)
        xh = float(match.group(1))
        xhlookup[z] = xh
    return xhlookup


def restrict_redshifts(longdict, zmin=0.0, zmax=1000.0):
    """
    From a dictionary whose keys are redshift values
    Restrict range of redshifts, since makes little 
    sense to look at highest redshifts where little happens
    """
    zuse = []
    for z in sorted(longdict.keys()):
        if z < zmax and z > zmin:
            zuse.append(z)
    print zuse
    return zuse

def find_redshifts(longdict, zlist = [6.0, 8.0]):
    """
    From a dictionary whose keys are redshift values
    Find the redshifts that are closest to those in zlist
    and return them. Use simple search to find minimum difference
    between desired and available redshifts
    """

    zmax = np.max(longdict.keys())
    zmin = np.min(longdict.keys())

    zuse = []
    for z in sorted(zlist):
        zold = 0.0
        if z > zmax:
            print "%0.3f exceeds max value of %0.3f" % (z, zmax)
            zuse.append(zmax)
        elif z < zmin:
            print "%0.3f lowers than min value of %0.3f" % (z, zmin)
            zuse.append(zmin)
        else:
            for ztest in sorted(longdict.keys()):
                if abs(ztest - z) > abs(zold - z):
                    zuse.append(zold)
                    break
                zold = ztest
            
    return zuse

def output_c2ray(datacube, outfilename = "test.dat", dtype = np.float32):
    """
    Write out a numpy datacube in the c2ray format
    i.e. as binary with first three integers the dimensions of
    the box.

    By default assumes float output
    """
    box = np.insert(datacube, 0, datacube.shape)
    of = open(outfilename, "wb")
    of.write(box.astype(dtype))
    of.close()

def smoothbox(box, sigma = 0):
    """
    Smooth a box with a Gaussian kernal of size scale pixels
    """

    if sigma >0:
        return scipy.ndimage.filters.gaussian_filter(box, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    else:
        return box

def maptobox(x, DIM = 256):
    """
    Take a vector of pixel locations 3 x N and remap to lie inside of the box
    """

    y = np.mod(x, DIM)
    return y
