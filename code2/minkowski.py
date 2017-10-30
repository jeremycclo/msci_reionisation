"""
Code to call the Minkowski functional code of Buchert
"""
import subprocess
import numpy as np

new = True

def run_minkowski_frombox(datacube,
                  DIM = 256,
                  nbins = 1, low_threshold = 0.5, high_threshold = 0.5, smoothing = 0):
    """
    
    """
    if new:
        box = datacube.copy()
    else:
        box = np.insert(datacube, 0, datacube.shape)

    #write the input data file
    infile = "input_mink.dat"
    dtype = np.float32
    of = open(infile, "wb")
    of.write(box.astype(dtype))
    of.close()

    outfile = "output_mink.dat"

    #run the code
    run_minkowski_new(infile, outfile, DIM, nbins, low_threshold, high_threshold, smoothing)

    #Read in data
    data = np.loadtxt(outfile)
    threshold = data[:, 0]
    V0 = data[:, 1]
    V1 = data[:, 2]
    V2 = data[:, 3]
    V3 = data[:, 4]

    return threshold, V0, V1, V2, V3
    

def run_minkowski(infile, outfile = "output.dat",
                  DIM = 256,
                  nbins = 1, low_threshold = 0.5, high_threshold = 0.5, smoothing = 0):
    #new = True
    if new:
        run_minkowski_new(infile,outfile,DIM,nbins, low_threshold, high_threshold, smoothing)
    else:
        run_minkowski_old(infile,outfile,DIM,nbins, low_threshold, high_threshold, smoothing)
        

def run_minkowski_new(infile, outfile = "output.dat",
                  DIM = 256,
                  nbins = 1, low_threshold = 0.5, high_threshold = 0.5, smoothing = 0):
    """
    run ./beyond

    This is the Minkowski-3 code from Buchert

    Assumes that infile contains a cubic datacube in binary format

    Will output nbins values on interval [lo, high]
    """

    #Evaluate Minkowski functional at nbins different thresholds between
    #low_threshold and high_threshold
    #nbins = 1
    #low_threshold = 0.5
    #high_threshold = 0.5

    #integer ideally a power of 2 to allow for Gaussian smoothing
    #smoothing = 2

    #oversampling
    intervals = 2

    #Note that beyond calculates thresholds internally on basis of
    # [lo, high] with nbin+1 values. This is inconsistent with the
    # Readme, but stems from loops like for(j=0;j<=nbins;j++), which
    # count nbins+1 values. Hence subtract one from nbins when passing
    # to beyond to ensure more sensible meaning to nbins.
    # Done this way output should match np.linspace(lo, high, nbins) and
    # python conventions for loops

    #assemble command string
    cmd = "/Users/jpritcha/Documents/current/projects/Solene/MinkowskiFunctionals/beyond/"
    cmd = cmd + "beyond -x%d -y%d -z%d -b%i -l%f -h%f -m%i -s%i -i%s -o%s -N -t" % (DIM, DIM, DIM, np.max(nbins-1, 1), low_threshold, high_threshold, intervals, smoothing, infile, outfile)
    print cmd
    subprocess.call(cmd, shell=True)
    

def run_minkowski_old(infile, outfile = "output.dat",
                  DIM = 256,
                  nbins = 1, low_threshold = 0.5, high_threshold = 0.5, smoothing = 0):
    """
    run ./Minkowski

    This makes use of the modified (and I think older) version of the code
    received from Suman by way of Martina
    """

    #Evaluate Minkowski functional at nbins different thresholds between
    #low_threshold and high_threshold
    #nbins = 1
    #low_threshold = 0.5
    #high_threshold = 0.5

    #integer ideally a power of 2 to allow for Gaussian smoothing
    #smoothing = 0

    #Note that there is a slight inconsistency in how minkowski has been
    #hacked and the threshold values. Internally the thresholds are
    #calculated on interval [lo, high] with nbins+1 values, but
    #the output is hacked to output only [lo, high) & exclude the top bin.

    cmd = "/Users/jpritcha/Documents/current/projects/Solene/MinkowskiFunctionals/Minkowski/"
    cmd = cmd + "minkowski -x%d -y%d -z%d -b%i -l%f -h%f -m2 -s%i -i%s -o%s -N -f -c -t" % (DIM, DIM, DIM, nbins, low_threshold, high_threshold, smoothing, infile, outfile)
    
    print cmd
    subprocess.call(cmd, shell=True)



########################
# Simple script to make a Gaussian random field and output to a file
# in the format required by the Minkowski codes
########################

def make_test_box(DIM = 128):
    """
    Gaussian box to test Minkowski code.

    Produces a [DIM, DIM, DIM] box with values chosen from a unit Gaussian.
    First three integrers of the output binary file are the dimensions of
    the box.
    """

    #Gaussian random field
    box = npr.normal(size = [DIM, DIM, DIM])

    #write out in a form that minkowski can read
    #Needs dimensions of box as first three ints
    box = np.insert(box, 0, [DIM, DIM, DIM])
    outfilename = "test_gaussian.dat"
    dtype = np.float32
    of = open(outfilename, "wb")
    of.write(box.astype(dtype))
    of.close()
    

######################
# Theoretical prediction
#####################

def gaussian_theory(sigma = 1.0, box = None):
    """
    Evaluate analytic expressions for Gaussian minkowski functionals
    from Schmalzing and Buchert (1997) and Gleser+ (2006)

    if box is not None will normalise to a box
    """

    if box is not None:
        sigma = sqrt(np.var(box))
        sigma1 = sqrt(np.var(np.gradient(box)))
    else:
        sigma1 = 1.0

    #dimensionless threshold
    threshold = np.linspace(-4.0, 4.0, 40) * sqrt(sigma)
    u = threshold / sqrt(sigma)

    #lambda parameter
    xi = sigma * sigma
    xipp  = sigma1 * sigma1
    lam = sqrt(xipp / (6.0 * pi * xi))

    #now calculate the Minkowski functionals
    V0 = 0.5 - 0.5 * erf(u / sqrt(2.0))
    V1 = (2.0 / 3.0) * (lam / sqrt(2.0 * pi)) * np.exp(- u * u /2.0)
    V2 = (2.0 / 3.0) * (lam * lam / sqrt(2.0 * pi)) * u * np.exp(- u * u /2.0)
    V3 = (2.0 / 3.0) * (lam * lam * lam / sqrt(2.0 * pi)) * (u*u -1.0) * np.exp(- u * u /2.0)    
    return V0, V1, V2, V3


def output_plot(infile, box = None):
    """
    Example plot of Minkowski functionals
    """

    #Theory calculation
    #if box is not None:
    #    threshold, V0, V1, V2, V3 = gaussian_theory()

    #Read in data
    data = np.loadtxt(infile)

    threshold = data[:, 0]
    V0 = data[:, 1]
    V1 = data[:, 2]
    V2 = data[:, 3]
    V3 = data[:, 4]

    plt.figure()
    plt.subplot(221)
    plt.plot(threshold, V0)

    plt.subplot(222)
    plt.plot(threshold, V1)

    plt.subplot(223)
    plt.plot(threshold, V2)

    plt.subplot(224)
    plt.plot(threshold, V3)

    plt.show()
