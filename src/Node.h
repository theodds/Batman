#ifndef NODE_H
#define NODE_H

#include <RcppArmadillo.h>
#include "TreeHypers.h"

template <typename T> 
struct Node {

  T* left;
  T* right; 
  T* parent;
  
  const TreeHypers* tree_hypers;

  bool is_root; 
  bool is_leaf;
  int depth; // Must be 1 + depth(parent) unless it is root

  double lower; // 0 <= lower <= upper <= 1
  double upper; 

  int var;
  int group;
  double val;
  
  // Init
  Node<T>(const TreeHypers* tree_hypers_) {
    depth       = 0;
    left        = NULL; 
    right       = NULL;
    parent      = NULL;
    is_root     = true;
    is_leaf     = true;
    var         = 0;
    val         = 0.0;
    group       = 0;
    tree_hypers = tree_hypers_;
    lower       = 0.0;
    upper       = 1.0;
  }

  Node<T>(T* parent) {
    is_leaf      = true;
    is_root      = false;
    this->parent = parent;
    left         = NULL;
    right        = NULL;
    var          = 0;
    group        = 0;
    val          = 0.0;
    depth        = parent->depth + 1;
    tree_hypers  = parent->tree_hypers;
    lower        = 0.0;
    upper        = 1.0;
  }

  ~Node<T>() {
    if(!(left == NULL)) delete left;
    if(!(right == NULL)) delete right;
  }
  
  void DeleteLeaves();
  bool is_left();
  void GetLimits();
  void BirthLeaves();

};

template<typename T>
void Node<T>::DeleteLeaves() {
    delete left;
    delete right;
    left = NULL;
    right = NULL;
    is_leaf = true;
}

template<typename T>
bool Node<T>::is_left() {
  if(parent == NULL) return true;
  return (this == parent->left);
}

template<typename T>
double probability_node_birth(T* root) {
  return root->is_leaf ? 1.0 : 0.5;
}

template<typename T>
void leaves(T* x, std::vector<T*>& leafs) {
  if(x->is_leaf) {
    leafs.push_back(x);
  }
  else {
    leaves(x->left, leafs);
    leaves(x->right, leafs);
  }
}

template<typename T>
std::vector<T*> leaves(T* x) {
  std::vector<T*> leafs;
  leaves(x,leafs);
  return leafs;
}

// TODO: Might be a better idea to cache the limits to improve speed in the future
template<typename T>
void Node<T>::GetLimits() {
  T* y = static_cast<T*>(this);
  lower = 0.0;
  upper = 1.0;
  bool my_bool = y->is_root ? false : true;
  while(my_bool) {
    bool is_left = y->is_left();
    y = y->parent;
    my_bool = y->is_root ? false : true;
    if(y->var == var) {
      my_bool = false;
      if(is_left) {
        upper = y->val;
        lower = y->lower;
      }
      else {
        upper = y->upper;
        lower = y->val;
      }
    }
  }
}

template<typename T>
void Node<T>::BirthLeaves() {
  if(is_leaf) {
    T* node              = static_cast<T*>(this);
    left                 = new T(node);
    right                = new T(node);
    is_leaf              = false;
    arma::uvec group_var = tree_hypers->SampleVar();
    group                = group_var(0); 
    var                  = group_var(1); 
    
    GetLimits();
    val     = (upper - lower) * unif_rand() + lower;
  }
}

template<typename T>
void Node<T>::GenBelow() {
  double grow_prob = GrowProb(tree_hypers, depth);
  double u = unif_rand();
  if(u < grow_prob) {
    BirthLeaves();
    left->GenBelow();
    right->GenBelow();
  }
}

template<typename NodeType>
NodeType* rand(std::vector<NodeType*> ngb) {
  int N = ngb.size();
  arma::vec p = arma::ones<arma::vec>(N) / ((double(N)));
  int i = sample_class(p);
  return ngb[i];
}

template<typename NodeType>
std::vector<NodeType*> not_grand_branches(NodeType* tree) {
  std::vector<NodeType*> ngb;
  not_grand_branches(ngb, tree);
  return ngb;
}

template<typename NodeType>
void not_grand_branches(std::vector<NodeType*>& ngb, NodeType* node) {
  if(!node->is_leaf) {
    bool left_is_leaf = node->left->is_leaf;
    bool right_is_leaf = node->right->is_leaf;
    if(left_is_leaf && right_is_leaf) {
      ngb.push_back(node);
    }
    else {
      not_grand_branches(ngb, node->left);
      not_grand_branches(ngb, node->right);
    }
  }
}

template<typename NodeType>
void ResetSuffStat(NodeType* node) {
  node->ss.Reset();
  if(!(node->left == NULL)) ResetSuffStat(node->left);
  if(!(node->right == NULL)) ResetSuffStat(node->right);
}

template<typename T>
double TreeLoglik(T* node) {
  double out = 0.0;
  if(node->is_leaf) {
    out += log(1.0 - GrowProb(node->tree_hypers, node->depth));
  }
  else {
    out += log(GrowProb(node->tree_hypers, node->depth));
    out += TreeLoglik(node->left);
    out += TreeLoglik(node->right);
  }
  return out;
}

template<typename T>
double ForestLoglik(std::vector<T*>& forest) {
  double out = 0.0;
  for(int t = 0; t < forest.size(); t++) {
    out += TreeLoglik(forest[t]);
  }
  return out;
}

template<typename T>
arma::uvec get_var_counts(std::vector<T*>& forest) {
  
  int num_groups = forest[0]->tree_hypers->get_num_groups();
  int num_tree = forest.size();
  arma::uvec counts = arma::zeros<arma::uvec>(num_groups); 
  for(int t = 0; t < num_tree; t++) {
    get_var_counts(counts, forest[t]); 
  }
  return counts;
}

template<typename T>
void get_var_counts(arma::uvec& counts, T* node) {
  if(!node->is_leaf) {
    counts(node->group) = counts(node->group) + 1;
    get_var_counts(counts, node->left); 
    get_var_counts(counts, node->right); 
  }  
}

// /*Note: Because the shape of the Dirichlet will mostly be small, we sample from
// the Dirichlet distribution by sampling log-gamma random variables using the
// technique of Liu, Martin, and Syring (2017+) and normalizing using the
// log-sum-exp trick */
template<typename T>
void UpdateS(TreeHypers& tree_hypers, std::vector<T*>& forest) {
  
  // Get shape vector
  int num_group = tree_hypers.get_num_groups(); 
  arma::vec shape_up = tree_hypers.alpha / ((double)num_group) * arma::ones<arma::vec>(num_group);
  shape_up = shape_up + get_var_counts(forest); 
  
  // Sample unnormalized s on the log scale
  arma::vec logs = arma::zeros<arma::vec>(num_group); 
  for(int i = 0; i < shape_up.size(); i++) {
    logs(i) = rlgam(shape_up(i)); 
  }
  // Normalize s on the log scale, then store
  logs = logs - log_sum_exp(logs); 
  tree_hypers.set_log_s(logs); // This also sets s = exp(logs)
}


#endif
