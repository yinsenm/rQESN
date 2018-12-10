#ifndef _ESN_UTILS_H
#define _ESN_UTILS_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Cholesky>
#include <iostream>
#include <functional>
#include <cmath>
#include <time.h>

//using namespace Eigen;
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;
using Eigen::Lower;
using Eigen::Ref;
using Eigen::SelfAdjointView;
using Eigen::Map;
using Eigen::MappedSparseMatrix;

// typedef
typedef MappedSparseMatrix<double> MSpMat;
typedef SparseMatrix<double> SpMat;
typedef Map<MatrixXd> MapMatd;


// Compute X'WX where W is a diagonal matrix (input w is a vector)
MatrixXd XtWX(const MatrixXd&, const VectorXd&);

// compute X'X 
MatrixXd XtX(const MatrixXd&);

// compute XX' 
MatrixXd XXt(const MatrixXd&);

// compute tanh
MatrixXd tanh(const MatrixXd&);

// generate random sparse matrix
MatrixXd initSparseMat(unsigned int, unsigned int, double, double);

#endif // !_ESN_UTILS_H

