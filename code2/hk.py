"""
PYthon implementation of the HK algorithm. Based upon algorithm and code
described at
https://www.ocf.berkeley.edu/~fricke/projects/hoshenkopelman/hoshenkopelman.html

This is restricted to 2D and doesn't allow for periodic boundary conditions.
Seems fairly easy to generalise though
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

def hoshen_kopelman(box):
    """
    Given binary input file in 2D, return labelled points identifying
    clusters
    """

    mask = box.copy()
    
    #list of labels initially set to labels[x] = x i.e. each element in array
    # has a unique identifier
    labels = np.arange(mask.size)
    label = np.zeros(mask.shape)

    largest_label = 0;
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            
            if mask[x,y]:
                #find value in left and up, being careful about edges
                #Note that removing the if statements here, should allow
                #for periodic boundary conditions
                if x>0:
                    left = mask[x-1, y].astype(int)
                else:
                    left = 0
                if y>0:
                    up = mask[x, y-1].astype(int)
                else:
                    up = 0

                #Check for neighbours

                #next line counts which of left and up are non-zero
                check = (not not left) + (not not up)
                
                if check == 0:  #Neither a label above nor to the left.
                    #Make a new, as-yet-unused cluster label.
                    mask[x, y] = make_set(labels)

                elif check == 1:   # One neighbor, either left or up.
                    mask[x,y] = max(up, left)  #use whichever is non-zero

                elif check == 2:  #Neighbors BOTH to the left and above.
                    #Link the left and above clusters
                    mask[x,y] = union(left, up, labels)

                else:
                    raise Exception("Invalid value for check")
                
                #print labels

    #Now at the end relabel to ensure consistency i.e. that clusters are
    #numbered sequentially
    max_labels = mask.size / 2
    n_labels = max_labels
    new_labels = np.zeros(n_labels)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                x = find_simple(mask[i, j].astype(int), labels)
                if new_labels[x] == 0:
                    new_labels[0] += 1
                    new_labels[x] = new_labels[0]
                mask[i, j] = new_labels[x]
                    
    return mask


def find_simple(x, labels):
    
    while(labels[x] != x):
        x = labels[x]
        
    return x


def union(x, y, labels):
    #Make two labels equivalent by linking their respective chains of aliases
    labels[find_simple(x, labels)] = find_simple(y, labels)
    return find_simple(y, labels)


def find(x, labels):
    """
    Better version of find - NOT TESTED
    """
    y = x.copy()
    z = 0

    #print x
    #print "finding"
    #print labels

    #This recurses down the chain of aliases until it finds the

    while (labels[y] != y):
        y = labels[y]

    #Next bit collapses the chain upweighting all labels in a chain
    #to the highest label. This saves time, since in the future a new
    #cell belonging to the cluster will immediately be identified with
    #that label rather than having to recurse along a potentially long
    #chain to get the final label value

    while (labels[x] !=x):
        #print x, labels
        z = labels[x].copy()
        labels[x] = y
        x = z
        #print x, labels

    #print y
    #print "done"

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
        for j in range(mask.shape[1]):
            if mask[i, j]:
                N = (0 if i == 0 else mask[i-1][j])
                S = (0 if i == mask.shape[0]-1 else mask[i+1][j])
                E = (0 if j == mask.shape[1]-1 else mask[i][j+1])
                W = (0 if j == 0 else mask[i][j-1])

                assert( N==0 or mask[i][j]==N )
                assert( S==0 or mask[i][j]==S )
                assert( E==0 or mask[i][j]==E )
                assert( W==0 or mask[i][j]==W )

    print "Labelling checks out"

#####################################################
# Terminal call to run c-code
#####################################################
def run_hk_from_terminal(box):
    """
    Command line call to hk.x - handled by files
    First write box to test2d.txt and pass that as input to
    hk.x using command hk.x -f filename.
    Then read in the output, which is written to hk.out

    Clunky, but effective
    """

    #save the box
    filename = "test2d.txt"
    file = open(filename, "w")
    file.write("%d %d\n" % (box.shape[0], box.shape[1]))
    for i in range(box.shape[0]):
        for j in range(box.shape[1]):
            file.write("%d " % (box[i,j]))
        file.write("\n")
    file.close()

    subprocess.check_output(['./hk.x',"-f", filename])
    #subprocess.call(['./hk.x',"-f", filename])
    clusters = np.loadtxt("hk.out")
    return clusters
                
##########################################
# Analysis functions
##########################################

def cluster_sizes(clusters):
    """
    Returns number of cells in different size clusters
    0 = not occupied
    1 - N are cluster labels
    """

    #number of clusters plus remainder
    n_clusters = np.max(clusters.astype(int)) + 1

    v_clusters = np.zeros(n_clusters)

    for i in range(n_clusters):
        v_clusters[i] = np.sum(clusters == i)
        
    return v_clusters

def locate_largest_cluster(clusters):
    """
    pick out the largest cluster for easy plotting
    ignore the 0-index stuff
    """

    #Find cluster label of largest cluster ignoring 0-index stuff
    volumes = cluster_sizes(clusters)
    largest = np.where(volumes[1:] == np.max(volumes[1:]))[0][0] + 1
    print "largest cluster is ", largest
    mask = np.zeros(clusters.shape)

    mask[clusters>0] = 1
    mask[clusters == largest] = 2

    return mask

def find_spanning_cluster(clusters):
    """
    Look for any clusters that have pixels on any two parallel edges
    Will return o-index
    """

    edge1 = set(clusters[:, 0])
    edge2 = set(clusters[:, -1])
    spanningx = edge1.intersection(edge2)

    #print "x: ", spanningx

    edge1 = set(clusters[0, :])
    edge2 = set(clusters[-1, :])
    spanningy = edge1.intersection(edge2)

    #print "y: ", spanningy

    #combine for all spanning clusters
    spanning = spanningx.union(spanningy)

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
        order = volumes[int(max(spanning))] / float(box.sum())
    else:
        order = 0.0

    return volumes, spanning, order

def summary_clusters(box, clusters):
    #pretty display

    p = box.sum() / float(box.size)

    #check for spanning cluster
    find_spanning_cluster(clusters)
    
    plt.figure(figsize=(6, 2.2))
    plt.subplot(121)
    plt.imshow(box, cmap=plt.cm.gray)
    plt.title("Input box f=%0.2f" % (p))
    plt.subplot(122)
    plt.imshow(locate_largest_cluster(clusters))
    plt.title("Clusters N=%i" %(np.max(clusters)))

    #size distribution
    volumes = cluster_sizes(clusters)
    print volumes
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

    m = n = 100
    ntrials = 1

    box = np.zeros([m, n])

    for trial in range(ntrials):
        p = npr.uniform()

        #create a random matrix thresholded on p
        mask = (npr.uniform(size = box.shape) < p)
        box[mask] = int(1)

        #print box

        #run the algorithm
        clusters = hoshen_kopelman(box)

        check_labelling(clusters)

        #print probability and cluster output
        print p, clusters

        summary_clusters(box, clusters)


if __name__ == "__main__":
    main()
