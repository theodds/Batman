#ifndef MCMC_H
#define MCMC_H

#include "Node.h"
#include "TreeHypers.h"

template<typename NodeType, typename DataType>
void TreeBackfit(std::vector<NodeType*>& forest, 
                 DataType& data) {
  
  double MH_BD = 0.7; 
  double MH_PRIOR = 0.4;
  int num_trees = forest.size(); 
  for(int t = 0; t < num_trees; t++) {
    BackFit(data, forest[t]);
    if(unif_rand() < MH_PRIOR) {
      forest[t] = draw_prior(forest[t], data);
    }
    if(forest[t]->is_leaf || unif_rand() < MH_BD) {
      birth_death(forest[t], data);
    }
    else {
      change_decision_rule(forest[t], data);
    }
    UpdateParams(forest[t], data);
    Refit(data, forest[t]);
  }
}

template<typename NodeType, typename DataType>
void birth_death(NodeType* tree, const DataType& data) {
  double p_birth = probability_node_birth(tree);
  if(unif_rand() < p_birth) {
    // Rcpp::Rcout << "doing birth\n";
    node_birth(tree, data);
  }
  else {
    // Rcpp::Rcout << "doing death\n";
    node_death(tree, data);
  }
}

template<typename T>
T* birth_node(T* tree, double& leaf_node_probability) {
  std::vector<T*> leafs = leaves(tree);
  T* leaf = rand(leafs);
  leaf_node_probability = 1.0 / ((double)leafs.size());

  return leaf;
}

template<typename T>
T* death_node(T* tree, double& p_not_grand) {
  std::vector<T*> ngb = not_grand_branches(tree);
  T* branch = rand(ngb);
  p_not_grand = 1.0 / ((double)ngb.size());
  
  return branch;
}

template<typename NodeType, typename DataType>
void node_birth(NodeType* tree, const DataType& data) {
  
  // Rcout << "Sample Leaf";
  double leaf_probability = 0.0;
  NodeType* leaf = birth_node(tree, leaf_probability);
  
  // Rcout << "Compute prior";
  double leaf_prior = GrowProb(leaf->tree_hypers, leaf->depth);
  
  // Get likelihood of current state
  // Rcout << "Current likelihood"; 
  double ll_before = LogLT(tree, data);
  ll_before += log(1.0 - leaf_prior);
  
  // Get transition probability
  // Rcout << "Transition";
  double p_forward = log(probability_node_birth(tree) * leaf_probability);
  
  // Birth new leaves
  // Rcout << "Birth";
  leaf->BirthLeaves();
  
  // Get likelihood after
  // Rcout << "New Likelihood";
  double ll_after = LogLT(tree, data);
  ll_after += log(leaf_prior) + 
    log(1.0 - GrowProb(leaf->tree_hypers, leaf->right->depth)) + 
    log(1.0 - GrowProb(leaf->tree_hypers, leaf->left->depth));
  
  // Get probability of reverse transition
  // Rcout << "Reverse";
  std::vector<NodeType*> ngb = not_grand_branches(tree);
  double p_not_grand = 1.0/((double)(ngb.size()));
  double p_backward = log((1.0 - probability_node_birth(tree)) * p_not_grand);
  
  // Do MH
  double log_trans_prob = ll_after + p_backward - ll_before - p_forward;
  if(log(unif_rand()) > log_trans_prob) {
    leaf->DeleteLeaves();
    leaf->var = 0;
    leaf->group = 0;
    // Rcpp::Rcout << "Reject!";
  }
  else {
    // Rcpp::Rcout << "Accept!";
  }
}

template<typename NodeType, typename DataType>
void node_death(NodeType* tree, const DataType& data) {
  // Select branch to kill Children
  double p_not_grand = 0.0;
  NodeType* branch = death_node(tree, p_not_grand);
  
  // Compute before likelihood
  double leaf_prob = GrowProb(branch->tree_hypers, branch->depth);
  double left_prior = GrowProb(branch->tree_hypers, branch->depth+1);
  double right_prior = left_prior;
  double ll_before = LogLT(tree, data) + 
    log(1.0 - left_prior) + log(1.0 - right_prior) + log(leaf_prob);
  
  // Compute forward transition prob
  double p_forward = log(p_not_grand * (1.0 - probability_node_birth(tree)));
  
  // Save old leafs 
  // Do not delete (they are dangling, need to be handled at end)
  NodeType* left = branch->left; 
  NodeType* right = branch->right; 
  branch->left = NULL;
  branch->right = NULL; 
  branch->is_leaf = true;
  
  // Compute likelihood after
  double ll_after = LogLT(tree, data) + log(1.0 - leaf_prob);
  
  // Compute backwards transition
  std::vector<NodeType*> leafs = leaves(tree);
  double p_backwards = log(1.0 / ((double)(leafs.size())) * 
                           probability_node_birth(tree));
  
  // Do MH and fix dangles
  double log_trans_prob = ll_after + p_backwards - ll_before - p_forward;
  if(log(unif_rand()) > log_trans_prob) {
    branch->left = left;
    branch->right = right;
    branch->is_leaf = false;
    // Rcpp::Rcout << "Reject!";
  }
  else {
    delete left;
    delete right;
    // Rcpp::Rcout << "Accept!";
  }
}

template<typename NodeType, typename DataType>
NodeType* draw_prior(NodeType* tree, const DataType& data)
{
  // Compute loglik before
  NodeType* tree_0 = tree;
  double loglik_before = LogLT(tree, data);

  // Make new tree and compute loglik after
  NodeType* tree_1 = new NodeType(tree_0);
  tree_1->is_root = true;
  tree_1->parent = tree_1;
  tree_1->depth = 0;
  tree_1->GenBelow();
  double loglik_after = LogLT(tree_1, data);

  // Do MH
  if(log(unif_rand()) < loglik_after - loglik_before) {
    delete tree_0;
    tree = tree_1;
  }
  else {
    delete tree_1;
  }
  return tree;
}

template<typename NodeType, typename DataType>
void change_decision_rule(NodeType* tree, const DataType& data) {
  std::vector<NodeType*> ngb = not_grand_branches(tree);
  NodeType* branch = rand(ngb);
  
  // Calculate likelihood before proposal
  double ll_before = LogLT(tree, data);
  
  // save old split
  int old_var      = branch->var;
  int old_group    = branch->group;
  double old_val   = branch->val;
  double old_lower = branch->lower;
  double old_upper = branch->upper; 
  
  // Modify the branch
  arma::uvec group_var = tree->tree_hypers->SampleVar(); 
  branch->group = group_var(0); 
  branch->var = group_var(1); 
  branch->GetLimits();
  branch->val = (branch->upper - branch->lower) * unif_rand() + branch->lower;
  
  // Calculate likelihood after proposal
  double ll_after = LogLT(tree, data);
  
  // Do MH
  double log_trans_prob = ll_after - ll_before;
  
  if(log(unif_rand()) > log_trans_prob) {
    branch->var = old_var;
    branch->group = old_group;
    branch->val = old_val;
    branch->lower = old_lower;
    branch->upper = old_upper; 
  }
}

template<typename NodeType, typename DataType, typename HypersType>
void IterateGibbs(std::vector<NodeType*>& forest, 
                  DataType& data, 
                  HypersType& hypers,
                  TreeHypers& tree_hypers) {
  TreeBackfit(forest, data);
  UpdateHypers(hypers, forest, data); 
  if(forest[0]->tree_hypers->update_s) UpdateS(tree_hypers, forest);
  if(tree_hypers.update_alpha) tree_hypers.UpdateAlpha();
  Rcpp::checkUserInterrupt(); 
}


#endif
