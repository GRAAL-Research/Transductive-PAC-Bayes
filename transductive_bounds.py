""" 
Empirical experiments of 'PAC-Bayesian Theory for Transductive Learning' (Begin et al., 2014)
http://graal.ift.ulaval.ca/aistats2014/

This file contains the implementation transductive PAC-Bayesians bounds

Code authors: Pascal Germain, Jean-Francis Roy
Released under the Simplified BSD license 
"""

from math import exp, log, sqrt
import numpy as np
from scipy import optimize
from scipy.stats import hypergeom

from d_functions import new_transductive_divergence


def compute_transductive_complexity_term(d_function, m, N):   
    """ Compute the complexity term of transductive PAC-Bayesian bounds.    
    See Equation (6) of Begin et al. (2014) for details.
    
    d_function: A convex function D:[0,1]x[0,1] -> Reals 
    m: Training set size (labeled examples)
    N: Full sample size (labeled + unlabeled examples) """
    one_over_m = 1./m
    m_over_N = float(m)/N
    
    def inside_sum(K):
        if K == 0: return 1.
        K_over_N = K/float(N)
        return sum([ exp(hypergeom.logpmf(k, N, K, m) + m*d_function(k*one_over_m, K_over_N, m_over_N) ) for k in np.arange( max(0,K+m-N), min(m,K)+1 ) ])
    
    return max( [ inside_sum(K) for K in np.arange( 0, N+1 ) ] )

    
def compute_general_transductive_gibbs_bound(d_function, empirical_gibbs_risk, m, N, KLQP, delta=0.05, complexity_term=None):
    """ Transductive PAC-Bayesian bound obtained with a specified ''D-function''.    
    See Theorem 5 of Begin et al. (2014) for details.
    
    d_function: A convex function of the form [0,1]x[0,1]->Reals 
    empirical_gibbs_risk : Gibbs risk observed on the training set
    m: Training set size (labeled examples)
    N: Full sample size (labeled + unlabeled examples)
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    complexity_term : None (default), or a precomputed value of complexity term 
    """ 
    if complexity_term is None:
        complexity_term = compute_transductive_complexity_term(d_function, m, N)
        
    m_over_N = float(m)/N       
    right_hand_side = ( KLQP + log( complexity_term / delta ) ) / m
    
    f = lambda x: d_function(empirical_gibbs_risk, x, m_over_N) - right_hand_side
    return optimize.brentq(f, empirical_gibbs_risk, 1.-1e-9) 


def compute_corollary_7a_gibbs_bound(empirical_gibbs_risk, m, N, KLQP, delta=0.05):
    """ Transductive PAC-Bayesian bound proposed by Corollary 7(a)
    in the form presented in Theorem S12 Begin et al. (2014, Supplementary Material)
    
    empirical_gibbs_risk : Gibbs risk observed on the training set
    m: Training set size (labeled examples)
    N: Full sample size (labeled + unlabeled examples)
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """
    m_over_N = float(m)/N 
    complexity_term = 3 * log(m) * sqrt( m*(1.-m_over_N) )  
    return compute_general_transductive_gibbs_bound(new_transductive_divergence, empirical_gibbs_risk, m, N, KLQP, delta, complexity_term)


def compute_corollary_7b_gibbs_bound(empirical_gibbs_risk, m, N, KLQP, delta=0.05):
    """ Transductive PAC-Bayesian bound proposed by Corollary 7(b)
    of Begin et al. (2014)
    
    empirical_gibbs_risk : Gibbs risk observed on the training set
    m: Training set size (labeled examples)
    N: Full sample size (labeled + unlabeled examples)
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """ 
    m_over_N = float(m)/N 
    log_complexity_term = log( 3 * log(m) * sqrt( m*(1.-m_over_N) ) )
    return empirical_gibbs_risk + sqrt( ((1.-m_over_N)/(2*m)) * (KLQP + log_complexity_term - log(delta) ) )


def compute_derbeko_2007_gibbs_bound(empirical_gibbs_risk, m, N, KLQP, delta=0.05):
    """ Transductive PAC-Bayesian bound proposed by Derbeko (2007)
    of Begin et al. (2014)
    
    empirical_gibbs_risk : Gibbs risk observed on the training set
    m: Training set size (labeled examples)
    N: Full sample size (labeled + unlabeled examples)
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """   
    m_over_N = float(m)/N 
    log_complexity_term = 7 * log( m*(N+1) )
    return empirical_gibbs_risk + sqrt( ((1.-m_over_N)/(2*m-2)) * (KLQP + log_complexity_term - log(delta) ) )

