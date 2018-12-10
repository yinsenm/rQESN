#include "utils.h"

MatrixXd train_esn(const MatrixXd& Y, const MatrixXd& X, MatrixXd& W, MatrixXd& U, MatrixXd& Z, double lambda, double leakrate, double pca, bool quadInd) {
	const int n(X.rows());
	const int n_x(X.cols());
	const int n_y(Y.cols());
	const int n_h(W.cols());
	MatrixXd V;
	MatrixXd H(MatrixXd::Zero(n, n_h));
	MatrixXd h(MatrixXd::Zero(n_h, 1));
	VectorXd y(VectorXd::Zero(n_y));
	VectorXd x(VectorXd::Zero(n_x));

	for (int k = 0; k < n - 1; k++) {
		y = Y.row(k);
		x = X.row(k + 1);
		h = tanh(MatrixXd(W * h) + MatrixXd(U * x) + MatrixXd(Z * y)).array() * (leakrate)+h.array() * (1 - leakrate);
		H.row(k + 1) = h.adjoint();
	}

	if (quadInd == false) {
		V = (XtX(H).array() + lambda * MatrixXd::Identity(n_h, n_h).array()).matrix().llt().solve(H.adjoint() * Y).adjoint();
	}
	else {
		MatrixXd Hq(n, 2 * n_h);
		Hq << H, H.array().square().matrix();
		V = (XtX(Hq).array() + lambda * MatrixXd::Identity(2 * n_h, 2 * n_h).array()).matrix().llt().solve(Hq.adjoint() * Y).adjoint();
	}
	return V;
}

MatrixXd predict_esn(const MatrixXd& X, MatrixXd& W, MatrixXd& U, MatrixXd& Z, MatrixXd& V, double leakrate, VectorXd y_last, bool quadInd) {
	const int n(X.rows());
	const int n_x(X.cols());
	const int n_h(W.cols());
	const int n_y(V.rows());
	MatrixXd H(MatrixXd::Zero(n, n_h));
	MatrixXd h(MatrixXd::Zero(n_h, 1));
	MatrixXd hq(2 * n_h, 1);
	MatrixXd Yh(MatrixXd::Zero(n, n_y));
	VectorXd y_add = y_last;
	VectorXd y(VectorXd::Zero(n_y));
	VectorXd x(VectorXd::Zero(n_x));

	for (int k = 0; k < n; k++) {
		x = X.row(k);
		h = tanh(MatrixXd(W * h) + MatrixXd(U * x) + MatrixXd(Z * y_add)).array() * (leakrate)+h.array() * (1 - leakrate);
		if (quadInd == false) {
			y_add = V * h;
		}
		else {
			hq << h, h.array().square().matrix();
			y_add = V * hq;
		}
		Yh.row(k) = y_add;
	}
	return Yh;
}
