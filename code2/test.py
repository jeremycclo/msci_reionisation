# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:19:50 2017

@author: ccl114
"""


import generate_gaussianRF as gGRF
import numpy as np
import matplotlib.pyplot as plt
import granulometry as gran


print 'my name is jlo'

def find_spanning_cluster(clusters):
    """
    Look for any clusters that have pixels on any two parallel edges
    Will return 0-index as well as

    Use set functionality to look for the intersection of points of edge1
    and edge 2 to find any clusters that are present at both edges. Since
    clusters are continguous this must mean those clusters span the space.
    """

    side = clusters.shape[0] * clusters.shape[1]
    edge1 = set(clusters[:, :, 0].reshape(side))
    edge2 = set(clusters[:, :, -1].reshape(side))
    spanningz = edge1.intersection(edge2)

    #print "z: ", spanningz

    edge1 = set(clusters[:, 0, :].reshape(side))
    edge2 = set(clusters[:, -1, :].reshape(side))
    spanningx = edge1.intersection(edge2)

    #print "x: ", spanningx

    edge1 = set(clusters[0, :, :].reshape(side))
    edge2 = set(clusters[-1, :, :].reshape(side))
    spanningy = edge1.intersection(edge2)

    #print "y: ", spanningy

    #combine for all spanning clusters
    spanning = spanningx.union(spanningy)
    spanning = spanning.union(spanningz)

    #print "spanning cluster is: ", spanning

    return spanning

def cluster_sizes(clusters):
    """
    Returns number of cells in different size clusters
    0 = not occupied
    1 - N are cluster labels

    Output has v_clusters[cluster_index] = number of pixels in that cluster
    """
    v_clusters = np.bincount(clusters.reshape(clusters.size))

    return v_clusters


def summary_statistics(box, clusters):
    """
    box is a binary box
    clusters is output of HK algorithm containing cluster labels
    
    Calculate volume distribution, idenity of spanning cluster
    and order parameter

    Order parameter is defined as
    (no. pixels in spanning cluster) / (no. pixels in all clusters)
    """
    volumes = cluster_sizes(clusters) # calling cluster_sizes function
    spanning = find_spanning_cluster(clusters)

    if max(spanning) > 0:
        order = volumes[max(spanning)] / float(box.sum()) # vol of largest perc cluster/ volme of box
    else:
        order = 0.0 # P_inf = 0 if there is no perc cluster

    return volumes, spanning, order


def main(box, clusters):
    volumes, spanning, order = summary_statistics(box, clusters)
    
    
    
    somthing_useful = 'somthing_useful'
    return something_useful
    
if __name__ == '__main__':
    
    
    


    field = gGRF.gaussian_random_field(Pk = gaussian(200.), size = 256)
    
    fieldB = bin
    
    
    plt.figure()
    plt.title('haha =', 10)
    plt.plot(np.arange(1, 5), np.arange(11, 15))
    
    plt.show()




#==============================================================================
#     plt.subplot(122)
#     hist, bin_edges = np.histogram(volumes[1:], bins = np.arange(int(np.max(volumes[1:])+3)))
#     plt.plot(bin_edges[0:-1], hist, 'ko-',linewidth=2,drawstyle='steps-mid')
#     plt.ylim([0.1, max(1, np.max(hist)+1)])
#     plt.ylabel("Number of clusters")
#     plt.xlabel("Size of cluster")
#     plt.title("Max size=%i" % (np.max(volumes[1:])))
#     plt.show()
#     plt.close()
#==============================================================================
    
    
    
    