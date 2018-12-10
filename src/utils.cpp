#include "utils.h"

// Compute X'WX where W is a diagonal matrix (input w is a vector)
MatrixXd XtWX(const MatrixXd& x, const VectorXd& w) {
	const int p(x.cols());
	MatrixXd AtWA(MatrixXd(p, p).setZero().
		selfadjointView<Lower>().rankUpdate(x.adjoint() * w.array().sqrt().matrix().asDiagonal()));
	return (AtWA);
}

// compute X'X 
MatrixXd XtX(const MatrixXd& x) {
	const int p(x.cols());
	MatrixXd AtA(MatrixXd(p, p).setZero().
		selfadjointView<Lower>().rankUpdate(x.adjoint()));
	return (AtA);
}

// compute XX' 
MatrixXd XXt(const MatrixXd& x) {
	const int n(x.rows());
	MatrixXd AAt(MatrixXd(n, n).setZero().
		selfadjointView<Lower>().rankUpdate(x));
	return (AAt);
}

// compute tanh
MatrixXd tanh(const MatrixXd& x) {
	const int n(x.rows()), p(x.cols());
	MatrixXd expmat(MatrixXd(n, p));  // exponential of matrix X
	expmat = (2 * x.array()).array().exp();
	expmat = (expmat.array() - 1) / (expmat.array() + 1);
	return (expmat);
}

// generate sparse matrix
MatrixXd initSparseMat(unsigned int nrow, unsigned int ncol, double w, double p) {
	// w is uniform boundary 
	// p is sparsity level
	MatrixXd W(MatrixXd::Random(nrow, ncol) * w);
	MatrixXd Mask = (MatrixXd::Random(nrow, ncol).cwiseAbs().array() < p).cast<double>();
	return Mask.cwiseProduct(W);
}
