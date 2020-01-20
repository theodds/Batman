TODO: 

1. Need to make the computations more efficient for the log-likelihood 
   computations. Currently, a full pass is taken on the log-likelihood of each
   tree at each iteration multiple times: we compute the before-likelihood, 
   we compute the after-likelihood, and we also do a full pass when the 
   parameters are updated. 
   
   This amount of computation is unnecessary. We really only need to compute
   a partial update associated to the tree modification at each iteration. This
   should not require doing any computation for the existing tree, while the 
   proposed tree should only require an update when we apply the birth move. 


2. Need to correct the SharedForest code so that the hyperparameters are 
   correctly updated. Currently, I think there is a bug in how these are being
   updated which leads to nonsense.
   
3. Need to fix initializations of more-or-less all the classes, but especially
   the ones related to the regression application.

4. There may be a way to increase speed by having UpdateParams called during
   LogLT. In all the models so far, the computation for these two functions are
   more-or-less the same, so it is wasteful to compute.
   
5. For the models currently written, things only make sense if we initialize the
   location (lambda_0 or mu_0) to be zero. This should be fixed in the future.
   The validity of the whole tree system depends on the sum of the values of the
   tree prediction PLUS LAMBDA_0 to be equal to lambda_hat, but this isn't
   enforced anywhere. 
   
6. Need to write UpdateHypers for all Poisson and MLogit models, and test that
   they work.


