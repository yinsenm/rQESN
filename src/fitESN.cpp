#include "utils.h"
#include "QESN.h"


// [[Rcpp::depends(RcppEigen)]]
typedef Map<MatrixXd> MapMatd;
typedef Map<VectorXd> MapVecd;
using Rcpp::List;
using Rcpp::Named;
using Rcpp::as;

// test if linear alg works
// [[Rcpp::export]]
List test(MapMatd& X) {
  MatrixXd xtx, xxt, h;
  xtx = XtX(X);
  xxt = XXt(X);
  h = tanh(X);
  return List::create(Named("xtx") = xtx,
                      Named("xxt") = xxt,
                      Named("h") = h);
}

// [[Rcpp::export]]
MatrixXd fitESN(MapMatd& Xtrain, MapMatd& Ytrain, MapMatd& Xtest, List param) {
  
  // load data and set dimension
  const int n(Xtrain.rows());
  const int n_x(Xtrain.cols());
  const int n_y(Ytrain.cols());
  
  if(n != Ytrain.rows()) {
    throw std::invalid_argument("Training X size (n) doesn't match up with training Y size");
  }
  
  cout << "Observations (n): " << n << endl
       << "Features (n_x): " << n_x << endl
       << "Tasks (n_y): " << n_y << endl;
  
  
  // set parameters for generating sparse matrix W, U and Z
  double a_w = abs(as<double>(param["a_w"]));
  double p_w = as<double>(param["p_w"]);
  double a_u = abs(as<double>(param["a_u"]));
  double p_u = as<double>(param["p_u"]);
  double a_z = abs(as<double>(param["a_z"]));
  double p_z = as<double>(param["p_z"]);
  
  // set other ESN parameters
  double leakrate = as<double>(param["leakrate"]);
  double nu = as<double>(param["nu"]);   // control spectral radius
  double pca = as<double>(param["pca"]);
  double lambda = abs(as<double>(param["lambda"]));
  unsigned int n_h = abs(as<int>(param["n_h"]));
  unsigned int seed = abs(as<int>(param["seed"]));
  bool quadInd = as<bool>(param["quadInd"]);
  
  
  // check if parameters are correct
  if(p_w > 1 || p_w < 0 || p_u > 1 || p_u < 0 || p_z > 1 || p_z < 0) {
    throw std::invalid_argument("Sparsity level parameter: p_w, p_u, and p_z should be within 0 and 1.");
  }
  
  if(leakrate > 1 || leakrate < 0) {
    throw std::invalid_argument("leakrate should be within 0 and 1.");
  }
  
  if(pca > 1 || pca < 0) {
    throw std::invalid_argument("precentage keeped in H (pca) should be within 0 and 1.");
  }
  
  if(nu > 1 || nu < 0) {
    throw std::invalid_argument("spectral radius (nu) should be within 0 and 1.");
  }
  
  cout << "Sparse parameters are set to be:" << endl 
       << "W: " << a_w << ", " << p_w << endl 
       << "U: " << a_u << ", " << p_u << endl 
       << "Z: " << a_z << ", " << p_z << endl << endl;
  
  cout << "Other parameters are set to be:" << endl 
       << "seed (seed): " << seed << endl
       << "hidden units (n_h): " << n_h << endl 
       << "leakrate (leakrate): " << leakrate << endl 
       << "spectral radius (nu): " << nu << endl 
       << "precentage keeped in H (pca): "<< pca << endl
       << "ridge penalty (lambda): "<< lambda << endl
       << "Use quad or not (quadInd): "<< quadInd << endl << endl;
  
  // generated W, U and Z sparse matrix
  std::srand((unsigned int) seed);
  MatrixXd W = initSparseMat(n_h, n_h, a_w, p_w);
  MatrixXd U = initSparseMat(n_h, n_x, a_u, p_u);
  MatrixXd Z = initSparseMat(n_h, n_y, a_z, p_z);
  
  double spectralRadius = W.eigenvalues().cwiseAbs().maxCoeff();       // maximum eigen value of W
  W = W * nu / spectralRadius;                                        // scaled W where deltaESN is the spectral radius
  cout << "Spectral Radius of W: " << spectralRadius << endl;
  
  clock_t t_beg, t_end; // time the training and testing
  t_beg = clock();
  cout << "begin training..." << endl;
  MatrixXd V = train_esn(Ytrain, Xtrain, W, U, Z, lambda, leakrate, pca, quadInd);
  
  cout << "begin predict..." << endl << endl;
  MatrixXd Yh = predict_esn(Xtest, W, U, Z, V, leakrate, Ytrain.row(n-1), quadInd);
  t_end = clock();
  double dt = (double)(t_end - t_beg) / (CLOCKS_PER_SEC);
  cout << "take " << dt << " seconds." << endl;
  
  return Yh;  
}
