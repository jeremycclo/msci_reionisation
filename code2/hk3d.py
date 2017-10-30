"""
PYthon implementation of the HK algorithm. Based upon algorithm and code
described at
https://www.ocf.berkeley.edu/~fricke/projects/hoshenkopelman/hoshenkopelman.html

This is restricted to 2D and doesn't allow for periodic boundary conditions.
Seems fairly easy to generalise though

Currently involved two loops through the data
1) to ID clusters
2) to ensuer labels are sequential
(optional 3rd pass to verify labels are correct)

Hence scales with box size as O(N^3) for N pixels on side of box. More importantly
though there's also an overhead to having lots of clusters

"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

verbose = False

def hoshen_kopelman(box):
    """
    Given binary input file in 3D, return labelled points identifying
    clusters
    """

    mask = box.copy()
    
    #list of labels initially set to labels[x] = x i.e. each element in array
    # has a unique identifier
    labels = np.arange(mask.size)
    label = np.zeros(mask.shape)

    largest_label = 0;
    for x in range(mask.shape[0]):

        if verbose:
            print ("ID clusters: %i of %i" % (x, mask.shape[0]) )

        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):            
                if mask[x,y,z]:
                    #find value in left and up, being careful about edges
                    #Note that removing the if statements here, should allow
                    #for periodic boundary conditions
                    if x>0:
                        left = mask[x-1, y, z].astype(int)
                    else:
                        left = 0
                    if y>0:
                        up = mask[x, y-1, z].astype(int)
                    else:
                        up = 0
                    if z>0:
                        behind = mask[x, y, z-1].astype(int)
                    else:
                        behind = 0

                    #Check for neighbours

                    #next line counts which of left and up are non-zero
                    check = (not not left) + (not not up) + (not not behind)

                    if check == 0:  #No occupied neighbours
                        #Make a new, as-yet-unused cluster label.
                        mask[x,y,z] = make_set(labels)

                    elif check == 1:   # One neighbor, either left or up.
                        #use whichever is non-zero
                        mask[x,y,z] = max(max(up, left), behind)

                    elif check == 2:  #Two neighbours to be linked together
                        if up and left:
                            mask[x,y,z] = union(left, up, labels)
                        elif up and behind:
                            mask[x,y,z] = union(behind, up, labels)
                        elif left and behind:
                            mask[x,y,z] = union(behind, left, labels)
                        else:
                            raise Exception("Something's gone wrong!")
                            
                    elif check == 3:  #Three neighbours to be linked
                        #Link all three via two unions of two
                        #Is ordering of this important? Doesn't seem to be
                        mask[x,y,z] = union(left, up, labels)
                        mask[x,y,z] = union(left, behind, labels)

                    else:
                        raise Exception("Invalid value for check")

    #Now at the end relabel to ensure consistency i.e. that clusters are
    #numbered sequentially

    #In 3D a checker pattern gives the largest number of possible clusters
    max_labels = mask.size / 2
    n_labels = max_labels
    new_labels = np.zeros(n_labels)

    for i in range(mask.shape[0]):

        if verbose:
            print ("Resequencing: %i of %i" % (i, mask.shape[0]) )
            
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k]:
                    x = find(mask[i, j, k].astype(int), labels)
                    if new_labels[x] == 0:
                        new_labels[0] += 1
                        new_labels[x] = new_labels[0]
                    mask[i, j, k] = new_labels[x]
                    
    return mask


def find_simple(x, labels):
    
    while(labels[x] != x):
        x = labels[x]
        
    return x


def union(x, y, labels):
    #Make two labels equivalent by linking their respective chains of aliases
    labels[find(x, labels)] = find(y, labels)
    return find(y, labels)


def find(x, labels):
    """
    Better version of find - Much faster
    """
    y = x.copy()
    z = 0

    while (labels[y] != y):
        y = labels[y]

    #second part collapses labels
    while (labels[x] !=x):
        z = labels[x].copy()
        labels[x] = y
        x = z

    return y

def make_set(labels):
    """
    Create a new equivalence class and return its class label
    """
    labels[0] += 1
    labels[labels[0]] = labels[0]
    return labels[0].astype(int)

def check_labelling(mask):
    """
    Check identification of clusters i.e. that all neighbours of a pixel
    have the same label
    """

    for i in range(mask.shape[0]):
        print ("Checking labels: %i of %i" % (i, mask.shape[0]) )
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i, j, k]:
                    N = (0 if i == 0 else mask[i-1][j][k])
                    S = (0 if i == mask.shape[0]-1 else mask[i+1][j][k])
                    E = (0 if j == mask.shape[1]-1 else mask[i][j+1][k])
                    W = (0 if j == 0 else mask[i][j-1][k])
                    U = (0 if k == 0 else mask[i][j][k-1])
                    D = (0 if k == mask.shape[2]-1 else mask[i][j][k+1])


                    assert( N==0 or mask[i][j][k]==N )
                    assert( S==0 or mask[i][j][k]==S )
                    assert( E==0 or mask[i][j][k]==E )
                    assert( W==0 or mask[i][j][k]==W )
                    assert( U==0 or mask[i][j][k]==U )
                    assert( D==0 or mask[i][j][k]==D )
                
    print ("Labelling checks out")


##########################################
# Analysis functions
##########################################

def cluster_sizes(clusters):
    """
    Returns number of cells in different size clusters
    0 = not occupied
    1 - N are cluster labels

    Output has v_clusters[cluster_index] = number of pixels in that cluster
    """
    v_clusters = np.bincount(clusters.reshape(clusters.size))

    return v_clusters

def locate_largest_cluster(clusters):
    """
    pick out the largest cluster for easy plotting
    ignore the 0-index stuff

    Output labels all clusters as 1 and largest cluster as 2 for easy
    plotting
    """

    #Find cluster label of largest cluster ignoring 0-index stuff
    volumes = cluster_sizes(clusters)
    largest = np.where(volumes[1:] == np.max(volumes[1:]))[0][0] + 1
    print ("largest cluster is ", largest)
    mask = np.zeros(clusters.shape)

    mask[clusters>0] = 1
    mask[clusters == largest] = 2

    return mask

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

def summary_statistics(box, clusters):
    """
    box is a binary box
    clusters is output of HK algorithm containing cluster labels
    
    Calculate volume distribution, idenity of spanning cluster
    and order parameter

    Order parameter is defined as
    (no. pixels in spanning cluster) / (no. pixels in all clusters)
    """
    volumes = cluster_sizes(clusters)
    spanning = find_spanning_cluster(clusters)

    if max(spanning) > 0:
        order = volumes[max(spanning)] / float(box.sum())
    else:
        order = 0.0

    return volumes, spanning, order

def size_distribution(volumes, nbins = 20):
    """
    Standardise construction of size histogram

    Sizes can become huge so need to assemble this carefully and using
    log bins
    """

    if len(volumes) == 1:
        #Handle case that there are no clusters at all
        return [], []

    #linear bins - becomes very slow if have large clusters
    bins = np.arange(int(np.max(volumes[1:])+3))

    #log bins
    delta = ( np.log(np.max(volumes[1:])) - np.log(1) ) / float(nbins-1)
    bins = np.exp( np.arange(nbins) * delta )
    hist, bin_edges = np.histogram(volumes[1:], bins = bins)

    return bin_edges[0:-1], hist
                                   

def summary_clusters(box, clusters):
    #pretty display

    p = box.sum() / float(box.size)

    print ("HK found %i clusters" % (np.max(clusters)) )

    MID = box.shape[0]/2

    volumes = cluster_sizes(clusters)
    print ("Largest cluster has size = ", np.max(volumes[1:]) )
    
    spanning = find_spanning_cluster(clusters)
    if max(spanning) > 0:
        print ("Spanning cluster exists and has size =", volumes[max(spanning)] )
        print ("Number of ionized pixels is =", box.sum() )
        print ("Order parameter is ", volumes[max(spanning)] / float(box.sum()) )

    #print clusters[:,:,MID]
    
    plt.figure(figsize=(6, 2.2))
    plt.subplot(121)
    plt.imshow(box[:,:,MID], cmap=plt.cm.gray)
    plt.title("Input box f=%0.2f" % (p))
    plt.subplot(122)
    plt.imshow(locate_largest_cluster(clusters)[:,:,MID])
    plt.title("Clusters N=%i" %(np.max(clusters)))

    #size distribution
    plt.figure()
    plt.subplot(121)
    plt.plot(volumes[1:], 'ko-',linewidth=2,drawstyle='steps-mid')
    plt.ylabel("Size of cluster")
    plt.xlabel("Cluster label")

    plt.subplot(122)
    hist, bin_edges = np.histogram(volumes[1:], bins = np.arange(int(np.max(volumes[1:])+3)))
    plt.plot(bin_edges[0:-1], hist, 'ko-',linewidth=2,drawstyle='steps-mid')
    plt.ylim([0.1, max(1, np.max(hist)+1)])
    plt.ylabel("Number of clusters")
    plt.xlabel("Size of cluster")
    plt.title("Max size=%i" % (np.max(volumes[1:])))
    plt.show()
    plt.close()


def main():
    """
    Test example with random field
    """

    import matplotlib.pyplot as plt

    reset = True

    m = n = l = 10
    ntrials = 1

    box = np.zeros([m, n, l])

    for trial in range(ntrials):
        p = npr.uniform()

        
        #create a random matrix thresholded on p
        if reset:
            mask = (npr.uniform(size = box.shape) < p)
            box[mask] = int(1)

        if reset:
            filename = "temp.dat"
            file = open(filename, "wb")
            file.write(box.astype(int))
            file.close()
        else:
            filename = "temp.dat"
            f = open(filename, "rb")
            dtype = np.int64
            data = f.read() 
            f.close()
            DIM = m
            _data = np.fromstring(data, dtype)
            _data.shape = (DIM, DIM, DIM)
            _data = _data.reshape((DIM, DIM, DIM), order='F')
            box = _data

        #print box

        #run the algorithm
        clusters = hoshen_kopelman(box)
        check_labelling(clusters)
        
        #print probability and cluster output
        #print p, clusters
        print ("prob = %f" % (p) )

        summary_clusters(box, clusters)


if __name__ == "__main__":
    main()
