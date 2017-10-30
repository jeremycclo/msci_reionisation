"""

Granulometry class for calculating bubble size distribution from 21cmFast
boxes

Example granulometry code from SciPy Lectures

http://www.scipy-lectures.org/advanced/image_processing/auto_examples/plot_granulo.html

There are other examples there with image segmentation and other techniques

"""

import numpy as np
import numpy.random as npr
from scipy import ndimage
import matplotlib.pyplot as plt

import os
import sys, getopt

##########################################################
# Class for producing box containing randomly sized disks
##########################################################
#
# Currently doesn't handle periodic boundary conditions
# Can handle uniformly distributed disks or disk distributed
# to ensure no overlap with other disks or the edges. This latter
# case is useful for testing.
#
class RandomDisks:
    """
    PRoduce a random box filled with disks
    """

    def __init__(self, DIM=256, fillingfraction = 0.5, params = [10.0, 2.0], NDIM = 2, nooverlap = True):
        """
        Initialise a DIM x DIM box with random bubbles until
        fillingfraction of pixels have been labelled

        if NDIM =3 make a 3D box
        """

        self.NDIM = NDIM  #number of dimensions e.g. 2D or 3D
        self.DIM = DIM   #Number of cells per dimension
        self.nooverlap = nooverlap
        
        #print "initialising a %iD box with %i cells" %(self.NDIM, self.DIM)
        #print "filling faction is %f and nooverlap is %i" %(fillingfraction, nooverlap)

        if NDIM == 2:
            self.box = np.ones([DIM, DIM])
        elif NDIM ==3:
            self.box = np.ones([DIM, DIM, DIM])
        else:
            raise Exception("NDIM must be 2 or 3" % (NDIM))

        self.Rmean = params[0]
        self.sigR = params[1]

        self.bubbles = []
        self.sizes = []

        #Add bubbles to get to target filling fraction
        self.increase_filling_fraction(fillingfraction)


    def summary(self):
        """
        Update summary statistics
        """
        print "Number of bubbles in=", len(self.bubbles)
        print "box mean = ", self.box.mean()
        #self.sizes = np.array(self.sizes)
        overlap = np.sum(np.pi * np.power(np.array(self.sizes), 2.0))
        overlap /= float(np.power(self.DIM, 2.0))
        print "volume in, mean volume, overlap = %f, %f, %f" % (overlap, 1.0 - self.box.mean(), overlap/(1.0 - self.box.mean()))

        #Show the slice
        #plt.imshow(self.box)
        #plt.show()

        #assemble size PDF
        hist, bin_edges = np.histogram(np.array(self.sizes), bins=10, normed = True)
        bins = (bin_edges[0:-1] + bin_edges[1:])/2.0
        self.pdf = (bins, hist)
        

    def increase_filling_fraction(self, target):
        """
        Add randomly located and sized bubbles until reach
        desired filling fraction
        """

        if 1.0 - target > self.box.mean():
            print "Box already more ionised than target"
            return
        
        while self.box.mean() > 1.0 - target:

            #Gaussian distribution of bubble sizes
            R = self.size_distribution(self.Rmean, self.sigR)

            #uniform distribution for bubble locations
            if self.nooverlap:
                x = self.uniform_distribution_nooverlap(R)
            else:
                x = self.uniform_distribution(R)

            #Use mask to ID pixels in this bubble
            #mask = self.disk_mask(x, R)
            mask = self.bubble_mask(x, R)
            self.box[mask] = 0

            #Store bubbles so can assemble true size PDF
            #Important if number of bubbles is small
            self.bubbles.append(x)
            self.sizes.append(R)
            #print x, R
            #print self.box.mean()

        #self.summary()


    def location_distribution(self, R):
        """
        Uniform distribution for bubbles
        """
        return npr.randint(self.DIM, size = self.NDIM)

    def uniform_distribution(self, R):
        """
        Uniform distribution for bubbles
        """
        return npr.randint(self.DIM, size = self.NDIM)

    def uniform_distribution_nooverlap(self, R):
        """
        Uniform distribution for bubbles without overlap

        This tests for overlap with other bubbles and also with the edge of the box
        """

        if self.NDIM >2:
            raise Exception("No overlap not coded for 3D")

        #My handling of the loop ending criterea is very messy
        accept = False
        count = 0
        while accept is False:
            count += 1

            #Choose random center for disk
            x = npr.randint(self.DIM, size = self.NDIM)
            
            #test to see if this would lead to overlap with another bubble or edge
            if len(self.bubbles) == 0:
                #First bubble in box will always be accepted
                accept = True
                test = True
            else:
                #Test for overlap with another bubble
                border = 7  #extra spacing to be conservative 
                test = np.all(
                    np.linalg.norm(self.bubbles - x, axis=1) > np.array(self.sizes)+R+border)
                
                #Exclude overlap with edges of cube too
                border = 5
                if x[0] < (R+border) or (x[0] + R + border) > self.DIM:
                    test = False
                if x[1] < (R+border) or (x[1] + R + border) > self.DIM:
                    test = False
                    
                if test == False:
                    pass
                    #print x
                    #print R
                    #print np.linalg.norm(self.bubbles - x, axis=1)
                    #print np.array(self.sizes)+R+20
                    #print np.linalg.norm(self.bubbles - x, axis=1) > np.array(self.sizes)+R+20

            #As box fills up may get hard to place new disks, so eventually
            #give up and accept some degree of overlap
            if count % 1000 == 0:
                print "WARNING: Taking a long time to place bubbles"
                if count %10000 == 0:
                    print "GIVING UP AND ACCEPTING OVERLAP!"
                    test = True
                    
            if test == True:
                return x
            else:
                accept = False



    def size_distribution(self, Rmean = 10, sigR = 3):
        #Return a bubble size drawn from a random distribution
        #here it's Gaussian, but forced to be >=1 to avoid
        #zero size bubbles, which are pointless
        R = max(1, int(npr.normal(Rmean, sigR)))
        return R

    def bubble_mask(self, x, n):
        #wrapper to handle different dimensionality
        #NOT FULLY WRITTEN FOR 3D yet
        if (self.NDIM == 2):
            return self.disk_mask(x,n)
        elif(self.NDIM == 3):
            return self.sphere_mask(x,n)
        else:
            raise Exception ("NDIM is not 2 or 3")

    def disk_mask_slow(self, pos, n):
        #This gives pixels within n of the centre (x0, y0)
        #Here just a 2D circle
        #
        #Memory inefficient at present
        #Disks are truncated at boundaries

        #print n
        struct = np.zeros((self.DIM, self.DIM))
        x, y = np.indices((self.DIM, self.DIM))
        mask = (x - pos[0])**2 + (y - pos[1])**2 <= n**2        
        struct[mask] = 1
        return struct.astype(np.bool)

    def disk_mask(self, pos, n):
        #This gives pixels within n of the centre (x0, y0)
        #Here just a 2D circle
        #
        #Memory inefficient at present
        #Disks are truncated at boundaries

        #print n
        structsize = 2 * n + 5
        struct = np.zeros((structsize, structsize))
        x, y = np.indices((structsize, structsize))
        mask = (x - structsize/2)**2 + (y - structsize/2)**2 <= n**2
        struct[mask] = 1
        
        #Now work out coordinate shift to move centre to pos
        xmov = structsize/2 - pos[0]
        ymov = structsize/2 - pos[1]

        #recreate mask
        full_struct = np.zeros([self.DIM, self.DIM])
        full_struct[x + xmov,y + ymov] = struct
        
        return full_struct.astype(np.bool)

    def disk_mask_periodic(self, pos, n):
        #This gives pixels within n of the centre (x0, y0)
        #Here just a 2D circle, but in such a way that
        #periodic boundary conditions are respected

        #Create a disk at centre of array to allow for large n
        #without hitting boundaries
        struct = np.zeros((self.DIM, self.DIM))
        x, y = np.indices((self.DIM, self.DIM))
        mask = (x - self.DIM/2)**2 + (y - self.DIM/2)**2 <= n**2
        struct[mask] = 1

        #Now work out coordinate shift to move centre to pos
        xmov = self.DIM/2 - pos[0]
        ymov = self.DIM/2 - pos[1]

        #use np.roll to move center to pos respecting perodicity
        struct=np.roll(np.roll(struct,shift=-xmov,axis=0),shift=-ymov,axis=1)
        return struct.astype(np.bool)

    def sphere_mask_new(self, pos, n):
        #This gives pixels within n of the centre (x0, y0, z0)
        #Here just a 3D sphere
        #This is very slow as wastes lots of time on not needed positions

        #print n
        struct = np.zeros((self.DIM, self.DIM, self.DIM))
        x, y, z = np.indices((self.DIM, self.DIM, self.DIM))
        mask = (
            (x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2)<= n**2
        struct[mask] = 1
        return struct.astype(np.bool)

    def sphere_mask(self, pos, n):
        #This gives pixels within n of the centre (x0, y0, z0)
        #Here just a 3D sphere
        #This is very slow as wastes lots of time on not needed positions

        print n
        #Create a disk at centre of array to allow for large n
        #without hitting boundaries
        structsize = 2 * n + 5
        struct = np.zeros((structsize, structsize, structsize))
        x, y, z = np.indices((structsize, structsize, structsize))
        mask = (x - structsize/2)**2 + (y - structsize/2)**2 + (z - structsize/2)**2<= n**2
        struct[mask] = 1

        #Now work out coordinate shift to move centre to pos
        xmov = structsize - pos[0]
        ymov = structsize - pos[1]
        zmov = structsize - pos[2]

        #Now need a way of applying this structure to the box in
        #the right place


        #recreate mask
        #x = x + xmov
        #y = y + ymov
        #z = z + ymov
        #full_struct = np.zeros([self.DIM, self.DIM, self.DIM])
        #This next line doesn't work as can go over edges of box
        #check for this
        [np.min(x) % self.DIM, np.max(x) % self.DIM]
        #full_struct[x, y, z] = struct
        
        #use np.roll to move center to pos respecting perodicity
        #This is slow when box is large
        #full_struct = np.zeros([self.DIM, self.DIM, self.DIM])
        #full_struct[x ,y, z] = struct
        #full_struct=np.roll(full_struct, shift = [-xmov, - ymov, -zmov], axis = [0,1,2])

        return full_struct.astype(np.bool)

##########################################################
# Granulometry class
##########################################################

class Granulometry:
    """
    Granulometry class.

    Notes:
        1) Works on an isolated 2D cube. i.e. doesn't account for periodic boundary conditions
        would need to code periodic boundary conditions separately. Probably by tiling the
        cube and then averaging somehow?
        2) Not obvious this should be a class!
    """
    def __init__(self):
        pass

    #File access stuff that doesn't really belong here
    def load_binary_data(self, filename, dtype=np.float32): 
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

    def readtocmbox(self, filename, DIM=256):
        # read in the data cube located in 21cmFast/Boxes/delta_T*
        data1 = self.load_binary_data(filename)
        data1.shape = (DIM, DIM, DIM)
        data1 = data1.reshape((DIM, DIM, DIM), order='F')
        return data1

    #Basic granulometry code

    def disk_structure(self, n):
        #Thsi is the strucutre element for granulometry
        #Here just a 2D circle

        #print n
        struct = np.zeros((2 * n + 1, 2 * n + 1))
        x, y = np.indices((2 * n + 1, 2 * n + 1))
        mask = (x - n)**2 + (y - n)**2 <= n**2
        struct[mask] = 1
        return struct.astype(np.bool)


    def granulometry(self, data, sizes=None):
        #Simple granulometry analysis. The key step here is the
        # ndimage.binary_opening() command
        #data should be a 2D array

        #Largest bubble is DIM/2 in radius
        s = max(data.shape)
        if sizes == None:
            sizes = range(1, s/2, 1)

        #Next line actually carries out granulometry
        granulo = [ndimage.binary_opening(data, \
            structure=self.disk_structure(n)).sum() for n in sizes]

        #Renormalise to give fraction of total pixels in each radii bin
        granulo = np.array(granulo) / float(data.size)

        #Calculate the differential distribution too
        dFdR = np.zeros(len(granulo))
        dFdR[0:-1] = granulo[0:-1] - granulo[1:]
        
        return sizes, granulo, dFdR


    def threshold(self, data):
        """
        Threshold for map
        """
        return data.mean()

    def mask(self, data):
        """
        Apply threshold to calculate mask
        """

        #Thresholding operation
        mask = data < self.threshold(data)
        return mask

    def analyse_slice(self, data, bubsizes = np.arange(1, 30, 1)):
        """
        Calculate both F(<R) and dF(<R)/dR

        Stick to pixel units
        """

        #Thresholding operation
        mask = self.mask(data)
        
        #Next lines carry out the actual granulometry. This gives a
        #cumulative distribution of pixels on given scale
        #analysis here is picking up all pixels in bubbles of a given scale
        #This is Koki's F(<R)
        R, F, dFdR = self.granulometry(mask, sizes=bubsizes)
        
        return R, F, dFdR

    def analyse_box(self, data, bubsizes = np.arange(1, 30, 1)):
        """
        Average over a 3D bpx
        returns R, mean(F), mean(dFdR), var(F), var(dFdR)
        """
        assert data.size == len(data)**3

        #Average over all the slices in the box
        Rs = []
        Fs = []
        dFs = []
        #for i in range(len(data)):
        for i in range(2):
            print "slice = ", i
            slice = data[:, :, i]       
            R, F, dFdR = self.analyse_slice(slice, bubsizes)
            Rs.append(R)
            Fs.append(F)
            dFs.append(dFdR)

        Fs = np.array(Fs)
        dFs = np.array(dFs)

        meanF = np.mean(Fs, 0)
        varF = np.var(Fs, 0)

        meanDF = np.mean(dFs, 0)
        varDF = np.var(dFs, 0)

        print R, meanF, meanDF, varF, varDF

        return R, meanF, meanDF, varF, varDF
    

    def mean_binning(self, R, F, dF, step = 4):
        """
        From radii R and F(<R) rebin in groups of step 
        """
        meanR = []
        meanF = []
        meandF = []
        indx = 0
        i = 0
        Rsum = 0.0
        Fsum = 0.0
        dFsum = 0.0
        #this loop combines bins in groups of step to get R and dFdlnR averaged
        #over step bins
        #(bit untidy and probably a more efficient way to code this)
        while i + step < len(R):

            Rsum += R[i]
            Fsum += F[i]
            dFsum += dF[i]
            #print Rsum, Fsum, dFsum
            i += 1
            #Check whether to go to next output bin
            if (i) % step == 0:
                indx += 1
                meanR.append(Rsum/step) #average bubble radii
                meanF.append(Fsum)
                meandF.append(dFsum/step)  #Not sure if should sum or average here
                Rsum = 0.0
                Fsum = 0.0
                dFsum = 0.0

        return np.array(meanR), np.array(meanF), np.array(meandF)

    def summary_plot(self, slice):

        mask = self.mask(slice)
        #Now plot the results in a pretty fashion
        #plt.figure(figsize=(6, 2.2))
        plt.figure()

        #plt.subplot(121)
        plt.imshow(mask, cmap=plt.cm.gray)
        opened = ndimage.binary_opening(mask, structure=self.disk_structure(4))
        opened_more = ndimage.binary_opening(mask, structure=self.disk_structure(18))
        plt.contour(opened, [0.5], colors='b', linewidths=2)
        plt.contour(opened_more, [0.5], colors='r', linewidths=2)
        plt.axis('off')
        #plt.subplot(122)
        
        #Plot the cumulative F(<R) versus R
        #plt.plot(R, F, 'ok', ms=8)

        #Add a curve for the dF/dlnR
        #plt.plot(R, dFdR)

        #plt.subplots_adjust(wspace=0.02, hspace=0.15, top=0.95, bottom=0.15, left=0, right=0.95)

        plt.show()
        plt.close()

    def pdf_plot(self, step = 2):
        """
        PDF plot with binning in cells of step
        """
        meanR, meanF, meandF = G.mean_binning(R, F, dFdR, step)

        plt.figure()
        plt.plot(R, dFdR/np.sum(dFdR), 'ko-',linewidth=2,drawstyle='steps-mid')
    
        plt.xlabel("R [Mpc]")
        plt.ylabel("dF/dR")
    
        plt.show()

        plt.close()

    
def test():
    #21cm data

    #set up

    G = Granulometry()
    DIM = 256
    DIM = 512
    deltaL = 300 / 256.0
    filename = "data/delta_T_v3_no_halos_z007.70_nf0.515895_useTs0_zetaX2.0e+56_alphaX1.2_TvirminX3.0e+04_aveTb009.76_Pop2_256_300Mpc"
    #filename = "data/xH_nohalos_z007.70_nf0.515895_eff20.0_effPLindex0.0_HIIfilter1_Mmin6.1e+08_RHIImax20_256_300Mpc"
    #filename = "data/delta_T_v3_no_halos_z006.73_nf0.219233_useTs0_zetaX2.0e+56_alphaX1.2_TvirminX3.0e+04_aveTb003.32_Pop2_256_300Mpc"
    #box = G.readtocmbox(filename)
    #slice = box[:,:,DIM/2]

    #loop over N slices

    RD = RandomDisks(DIM=DIM, fillingfraction=0.1)
    slice = RD.box

    #masked map for interest
    mask = G.mask(slice)

    #analysis here is picking up all pixels in bubbles of a given scale
    bubsizes = np.arange(1, 15, 1)
    R, F, dFdlnR = G.analyse_slice(slice, bubsizes)
    #R, F, dFdlnR, sigF, sigDF = G.analyse_box(box)    

    #Now plot the results in a pretty fashion
    plt.figure(figsize=(6, 2.2))

    plt.subplot(121)
    plt.imshow(mask, cmap=plt.cm.gray)
    opened = ndimage.binary_opening(mask, structure=G.disk_structure(4))
    opened_more = ndimage.binary_opening(mask, structure=G.disk_structure(18))
    plt.contour(opened, [0.5], colors='b', linewidths=2)
    plt.contour(opened_more, [0.5], colors='r', linewidths=2)
    plt.axis('off')
    plt.subplot(122)

    #Plot the cumulative F(<R) versus R
    plt.plot(R, F, 'ok', ms=8)

    #Add a curve for the dF/dlnR
    plt.plot(R, dFdlnR)

    plt.subplots_adjust(wspace=0.02, hspace=0.15, top=0.95, bottom=0.15, left=0, right=0.95)


    print R, F

    #The PDF from a single slice is quite noisy, so let's combine some
    # N of the R bins to reduce the scatter
    #(would be better to average over slices)
    #

    step = 2
    meanR, meanF, meandF = G.mean_binning(R, F, dFdlnR, step)

    binwidth = step * deltaL

    print meanR, meanF, meandF
    deltaL = 1
    plt.figure()
    plt.plot(R, dFdlnR/np.sum(dFdlnR), 'ko-',linewidth=2,drawstyle='steps-mid')
    
    def gauss(x):
        R0 = 5
        sig = 2
        return np.exp(-(x-R0)**2/(2*sig*sig))/np.sqrt(2.0 * np.pi * sig)
    
    plt.plot(R * deltaL, gauss(R * deltaL))

    pdf = RD.pdf
    plt.plot(pdf[0], pdf[1]/np.sum(np.array(pdf[1])), drawstyle='steps-mid' )
    plt.xlabel("R [Mpc]")
    plt.ylabel("dF/dlnR")
    
    plt.show()

    plt.close()

# This is the standard boilerplate that calls the test() function.
if __name__ == '__main__':
    test()
