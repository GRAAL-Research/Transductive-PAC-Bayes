""" 
Empirical experiments of 'PAC-Bayesian Theory for Transductive Learning' (Begin et al., 2014)
http://graal.ift.ulaval.ca/aistats2014/

This file contains the implementation of various 'D-functions', 
i.e., convex functions D:[0,1]x[0,1] -> Reals

Code authors: Pascal Germain, Jean-Francis Roy
Released under the Simplified BSD license 
"""

from math import log 

def kl_divergence(q, p, *args):
    """ See Equation (3) of Begin et al. (2014)""" 
    if q == p: return 0.
    if q <= 0.: return -log(1.-p)
    if q >= 1.: return -log(p)
    
    if p < 1e-9 or p > 1.-1e-9: return 1e9 # Arbitrary big number     
    
    return q*log( q/p ) + (1.-q)*log( (1.-q)/(1.-p) )

def new_transductive_divergence(q, p, m_over_N, *args):
    """ See Equation (8) of Begin et al. (2014)""" 
    if q == p: return 0.
    
    if p+1e-9 < m_over_N * q or p-1e-9 > 1.-m_over_N * (1.-q): return 1e9 # Arbitrary big number   
    
    H = lambda x: -x * log(x) - (1.-x)*log(1.-x) if x > 1e-9 and x < 1-1e-9 else 0.  
    return ( -p * H( m_over_N*q/p )-(1.-p) * H( m_over_N*(1.-q)/(1.-p) ) + H( m_over_N ) ) / m_over_N 

def quadratic_distance(q, p, *args):
    """ See Section 3.1 of Begin et al. (2014)"""
    return 2 * (q-p)**2
    
def variation_distance(q, p, *args):
    """ See Section 3.1 of Begin et al. (2014)"""
    return 2 * abs(q-p)

def triangular_discrimination(q, p, *args):
    """ See Section 3.1 of Begin et al. (2014) """
    if q == p: return 0.
    
    return (q-p)**2/(q+p) + (q-p)**2/(2.-q-p)


