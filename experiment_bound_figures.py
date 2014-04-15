""" 
Empirical experiments of 'PAC-Bayesian Theory for Transductive Learning' (Begin et al., 2014)
http://graal.ift.ulaval.ca/aistats2014/

This file allows to reproduce Figures 1, 2 and 3 (the two latter are in Supplementary Material).
Please uncomment the lines in the 'main' function and play with parameters at your will!

Code authors: Pascal Germain, Jean-Francis Roy
Released under the Simplified BSD license 
"""

def main():   
    # Generate Figure 1 of Begin et al. (2014), without N=5000 for quicker calculations
    generate_bound_figures(N_list=[200,500], ratios_list=[0.1, 0.5, 0.9], risk=0.2, KLQP=5.0)

    # Generate the whole Figure 1 of Begin et al. (2014)
    #generate_bound_figures(N_list=[200,500,5000], ratios_list=[0.1, 0.5, 0.9], risk=0.2, KLQP=5.0)

    # Generate the whole Figure 2 of Begin et al. (2014, Supplementary Material)
    #generate_bound_figures(N_list=[200,500,5000], ratios_list=[0.1, 0.5, 0.9], risk=0.1, KLQP=5.0)
    
    # Generate the whole Figure 3 of Begin et al. (2014, Supplementary Material)
    #generate_bound_figures(N_list=[200,500,5000], ratios_list=[0.1, 0.5, 0.9], risk=0.01, KLQP=5.0)
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from collections import OrderedDict
from math import log
from matplotlib import pyplot

from transductive_bounds import compute_general_transductive_gibbs_bound, compute_transductive_complexity_term
import d_functions  

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def generate_bound_figures(N_list, ratios_list, risk, KLQP, delta=0.05):
    """ Illustrate of the bound calculations
    
    N_list : List of full sample size
    ratios_list : List of m/N rations ( where m is the number of labeled examples)
    risk : Gibbs risk observed on the training set
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """

    divergences_dict = OrderedDict()
    divergences_dict["Kullback-Leibler"] = d_functions.kl_divergence
    divergences_dict["D*-function"] = d_functions.new_transductive_divergence
    divergences_dict["Quadratic Distance"] = d_functions.quadratic_distance
    divergences_dict["Variation Distance"] = d_functions.variation_distance
    divergences_dict["Triangular Discrimination"] = d_functions.triangular_discrimination

    n_rows = len(ratios_list)
    n_cols = len(divergences_dict)
    x_values = np.arange(0., 1.0, .005) 

    pyplot.subplots_adjust(wspace=0.1, hspace=0.1)

    STATS_dict = dict()

    for i, divergence_name in enumerate(divergences_dict.keys()):
        print('*** D-function: ' + divergence_name + ' ***')
        
        for j, ratio in enumerate(ratios_list):
            ax = pyplot.subplot(n_rows, n_cols, j*n_cols + i + 1)

            # Compute and draw d-function values (blue curves)
            divergence = divergences_dict[divergence_name]
            divergence_values = [divergence(risk, x, ratio) for x in x_values]
            pyplot.plot(x_values, divergence_values, linewidth=2)

            # Compute and draw bound values (horizontal lines) for each value of N
            for N in N_list:
                m = N * ratio

                complexity_term = compute_transductive_complexity_term(divergence, m, N)
                bound = compute_general_transductive_gibbs_bound(divergence, risk, m, N, KLQP, delta=delta, complexity_term=complexity_term)
                rhs = (KLQP + log(complexity_term / delta)) / m
                print('m=%d N=%d bound=%0.3f' % (m,N,bound) )

                handle = pyplot.plot([-1., bound, 2.],  3*[rhs], 'o--', label='%0.3f' % bound)[0] 

                STATS_dict[(i, N, ratio)] = (bound, rhs, handle)

            # Compute and draw risk limits (vertical dashed lines)
            risk_sup = 1. - ratio * (1.-risk)
            risk_inf = ratio * risk
            pyplot.plot(2*[risk_sup],  [0., 1.], 'k:')
            pyplot.plot(2*[risk_inf],  [0., 1.], 'k:')

            # Set plot properties
            pyplot.legend(loc=2)
            pyplot.xlim(0., 1.)
            pyplot.ylim(0., .5 if ratio > .4 else 1.)
            
            if j == n_rows-1:
                pyplot.xlabel(divergence_name)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(13)
            else:
                pyplot.setp(ax.get_xticklabels(), visible=False)
                
            if i == 0:
                pyplot.ylabel("m/N = %0.1f" % ratio)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(13)
            else:
                pyplot.setp(ax.get_yticklabels(), visible=False)

    # Highlight lower bounds for each (m,N) pairs 
    for N in N_list:
        for j, ratio in enumerate(ratios_list):
            best_bound = 1e6
            best_i = -1
            for i, _ in enumerate(divergences_dict.keys()):
                bound, rhs, handle = STATS_dict[(i, N, ratio)]
                if bound < best_bound:
                    best_bound, best_handle, best_i = bound, handle, i

            best_handle.set_marker('*')
            best_handle.set_markersize(14.)
            best_handle.set_markeredgewidth(0.)
            pyplot.subplot(n_rows, n_cols, j * n_cols + best_i + 1)
            pyplot.legend(loc=2)

    pyplot.show()
    
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
if __name__ == '__main__':
    print( __doc__ ) 
    main()
    

    
