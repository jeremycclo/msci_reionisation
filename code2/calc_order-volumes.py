# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:10:19 2017

@author: fd1014
"""

from __future__ import print_function
import types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import granulometry
import hk
import generate_gaussianRF as grf
import timeit

matplotlib.rcParams.update({'font.size': 14, 'axes.labelsize': 16})
matplotlib.rc('font',family='Times New Roman')


def calc_order(distinct_runs, av_runs, box_size, model, x_i_range, gen_new = True, timed = True):
    
    """
    Calculate order parameter for multiple ionization fractions, with capacity
    to average over many runs.
    """
    
    if timed == True:
        start_time = timeit.default_timer()
    
    x_i = np.linspace(x_i_range[0], x_i_range[1], num = distinct_runs, endpoint = False)
    Pinf = [] # list of order parameter data for each x_i value, averaged over runs.
    vol_hist = [] # list of volumes data for each x_i value
    
    if gen_new == False:
        field = grf.generate(Pk = model, size = box_size)
        fieldR = field.real
        av_runs = 1
    
    for i in range(distinct_runs):
        order_param = []
        vols = []
        for j in range(av_runs):
            print('\rFilling fraction is %f (%i)' % (x_i[i], j+1), end = '')
    
            if type(model) == list:
                RD = granulometry.RandomDisks(DIM = box_size, fillingfraction = x_i[i], params = model, nooverlap = False)
                box_inv = RD.box
                box = np.ones(box_inv.shape) - box_inv # invert box so filled disks identified as 1's
            
            if type(model) == types.LambdaType:
                if gen_new == True:
                    field = grf.generate(Pk = model, size = box_size)
                    fieldR = field.real
                box = grf.binary(fieldR, x_i[i])
            
            clusters = hk.hoshen_kopelman(box)
            volumes, spanning, order = hk.summary_statistics(box, clusters)
            order_param.append(order)
            if len(volumes) > 1:
                vols = np.concatenate((vols,volumes[1:]))
        
        Pinf.append(np.mean(order_param))
        
        if len(vols) > 0:
            hist, bin_edges = np.histogram(vols, bins = np.logspace(0, np.log10(int(np.max(vols)+3)), 20))
            vol_hist.append([hist, bin_edges])
    
    elapsed = timeit.default_timer() - start_time
    print('\rCompleted in %f s' % elapsed, end = '')
    
    return x_i, Pinf, vol_hist
 
    
def all_plots(x_i_for_size_distr, model_name = False):
    
    # Order parameter
    plt.figure(figsize = (6, 6))
    plt.plot(x_i, Pinf)
    plt.xlabel(r'$x_i$')
    plt.ylabel(r'$\frac{P_\infty}{x_i}$')
    if model_name:
        plt.savefig('Pinf__runs%ix%i_size%i_model-%s_xi(%.2f-%.2f).png' 
                    % (distinct_runs,av_runs,box_size,model_name,x_i_range[0],x_i_range[1]))
    
    # Cluster volume distribution
    for x_i_size in x_i_for_size_distr:
        plt.figure(figsize = (6, 6))
        bin_edges = vol_hist[int(len(vol_hist)*x_i_size)][1]
        hist = np.asarray(vol_hist[int(len(vol_hist)*x_i_size)][0], dtype=float)
        counts = sum(hist)
        plt.plot(np.log10(bin_edges[0:-1]), np.log10(hist/counts), 'ko-', linewidth = 2, drawstyle = 'steps-mid')
        plt.xlabel(r'$\log(V)$')
        plt.ylabel(r'Frequency')
        if model_name:
            plt.savefig('vol_distr__runs%ix%i_size%i_model-%s_xi%.2f.png' 
                        % (distinct_runs,av_runs,box_size,model_name,x_i_size))
    
    plt.show()


# Parameters
"""

<model> can be:
    [mean, variance]              --> random disks with mean and variance parameters
    grf.power_law(exponent)       --> GRF with power law as power spectrum
    grf.gaussian(mean, variance)  --> GRF with gaussian power spectrum
    grf.expn(scale)               --> GRF with exponential power spectrum
    or other lambda function..    --> GRF with arbitrary function as power spectrum

"""
distinct_runs = 20
av_runs = 20
box_size = 256
model = grf.power_law(-2.)
x_i_range = [0., 1.]

# Simulate order paramter and cluster volume distribution
x_i, Pinf, vol_hist = calc_order(distinct_runs, av_runs, box_size, model, x_i_range, gen_new = True)

all_plots(x_i_for_size_distr = [0.25, 0.50, 0.75], model_name = 'grf.power_law(-2.)')
