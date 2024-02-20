#include "CLogLogOrdinalgNode.h"

using namespace arma;
using namespace Rcpp;

// void GammaNode::AddSuffStat(const GammaData& data, int i, double phi) {
//   ss.Increment(data.Y(i), data.lambda_hat(i), phi);
//   if(!is_leaf) {
//     double x = data.X(i,var);
//     if(x <= val) {
//       left->AddSuffStat(data, i, phi);
//     } else {
//       right->AddSuffStat(data,i, phi);
//     }
//   }
// }

// void GammaNode::UpdateSuffStat(const GammaData& data, double phi) {
//   ResetSuffStat();
//   int N = data.X.n_rows;
//   for(int i = 0; i < N; i++) {
//     AddSuffStat(data,i, phi);
//   }
// }

// double PredictPois(GammaNode* n, const rowvec& x) {
//   if(n->is_leaf) {
//     return n->lambda;
//   }
//   if(x(n->var) <= n->val) {
//     return PredictPois(n->left, x);
//   }
//   else {
//     return PredictPois(n->right, x);
//   }
// }

// arma::vec PredictPois(GammaNode* tree, const arma::mat& X) {
//   int N = X.n_rows;
//   vec out = zeros<vec>(N);
//   for(int i = 0; i < N; i++) {
//     rowvec x = X.row(i);
//     out(i) = PredictPois(tree, x);
//   }
//   return out;
// }

// void BackFit(GammaData& data, GammaNode* tree) {
//   vec lambda = PredictPois(tree, data.X);
//   data.lambda_hat = data.lambda_hat - lambda;
// }

// void Refit(GammaData& data, GammaNode* tree) {
//   vec lambda = PredictPois(tree, data.X);
//   data.lambda_hat = data.lambda_hat + lambda;
// }

// double LogLT(GammaNode* root, const GammaData& data) {
//   root->UpdateSuffStat(data, root->pois_params->get_phi());
//   std::vector<GammaNode*> leafs = leaves(root);

//   double out = 0.0;
//   int num_leaves = leafs.size();

//   for(int i = 0; i < num_leaves; i++) {
//     double sum_1_by_phi = leafs[i]->ss.sum_1_by_phi;
//     double sum_y_exp_eta = leafs[i]->ss.sum_exp_lambda_minus_y;
//     double alpha = root->pois_params->get_alpha();
//     double beta = root->pois_params->get_beta();
//     double alpha_up = alpha + sum_1_by_phi;
//     double beta_up = beta + sum_y_exp_eta;

//     out += beta * log(alpha) - R::lgammafn(alpha);
//     out += R::lgammafn(alpha_up) - beta_up * log(alpha_up);
//   }
//   return out;
// }

// void UpdateParams(GammaNode* root, const GammaData& data) {
//   root->UpdateSuffStat(data, root->pois_params->get_phi());
//   std::vector<GammaNode*> leafs = leaves(root);
//   int num_leaves = leafs.size();
//   for(int i = 0; i < num_leaves; i++) {
//     double sum_1_by_phi = leafs[i]->ss.sum_1_by_phi;
//     double sum_y_exp_eta = leafs[i]->ss.sum_exp_lambda_minus_y;
//     double alpha = root->pois_params->get_alpha();
//     double beta = root->pois_params->get_beta();
//     double alpha_up = alpha + sum_1_by_phi;
//     double beta_up = beta + sum_y_exp_eta;

//     leafs[i]->lambda = rlgam(alpha_up) - log(beta_up);
//   }
// }
