""" 
Empirical experiments of 'PAC-Bayesian Theory for Transductive Learning' (Begin et al., 2014)
http://graal.ift.ulaval.ca/aistats2014/

This file allows to compute values of PAC-Bayesians bounds presented in the paper (like in Table 1).
Please change the parameters in the 'main' function at your will!


Code authors: Pascal Germain, Jean-Francis Roy
Released under the Simplified BSD license 
"""

def main(): 
    m = 250
    N = 500
    risk = 0.2
    KLQP = 5.0
    disagreement = 0.35
    delta = 0.05
       
    compute_multiple_bounds(m, N, risk, KLQP, disagreement, delta)

    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import transductive_bounds as bounds
import d_functions  

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def compute_multiple_bounds(m, N, risk, KLQP, disagreement=None, delta=0.05):
    """ Compute bounds on the full sample and print their values.
    
    m: Training set size (labeled examples)
    N: Full sample size (labeled + unlabeled examples)
    risk : Gibbs risk observed on the training set
    KLQP : Kullback-Leibler divergence between prior and posterior
    disagreement : Expected disagreement on the full sample
    delta : confidence parameter (default=0.05)
    """ 
    
    print('*** Parameters ***')
    print('m = %d\nN = %d\nrisk = %f\nKLQP = %f\ndisagreement = %f\ndelta = %f\n'
            % (m, N, risk, KLQP, disagreement, delta) )
    
    print('*** Bounds on the risk of the Gibbs Classifier ***')

    thm5_KL = bounds.compute_general_transductive_gibbs_bound(d_functions.kl_divergence,risk, m, N, KLQP, delta)
    print('Theorem 5-KL  : %f' % thm5_KL)  
    
    thm5_Dstar = bounds.compute_general_transductive_gibbs_bound(d_functions.new_transductive_divergence, risk, m, N, KLQP, delta)
    print('Theorem 5-D*  : %f' % thm5_Dstar)  

    cor7a = bounds.compute_corollary_7a_gibbs_bound(risk, m, N, KLQP, delta)
    print('Corollary 7(a): %f' % cor7a)
    
    cor7b = bounds.compute_corollary_7b_gibbs_bound(risk, m, N, KLQP, delta)
    print('Corollary 7(b): %f' % cor7b)
    
    derbeko = bounds.compute_derbeko_2007_gibbs_bound(risk, m, N, KLQP, delta)
    print('Derbeko (2007): %f' % derbeko)   
   
    print('\n*** Bounds on the risk of the Gibbs Classifier (Based on Theorem 5-D*) ***')
    
    twice = 2 * thm5_Dstar
    print('Twice the Gibbs: %f' % twice)
    if disagreement is None:
        print('Via the C-Bound: [disagreement required]')
    else:
        c_bound = compute_c_bound(thm5_Dstar, disagreement)
        print('Via the C-Bound: %f' % c_bound)


def compute_c_bound(gibbs_risk, disagreement):
    if gibbs_risk >= .5:
        return 1.0
    else:
        return 1.0 - (1.0 - 2*gibbs_risk)**2 / (1.0 - 2*disagreement)

 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
if __name__ == '__main__':
    print( __doc__ ) 
    main()
    

    