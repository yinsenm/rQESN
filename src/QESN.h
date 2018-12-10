#ifndef _QESN_H
#define _QESN_H
#include "utils.h"

MatrixXd train_esn(const MatrixXd&, const MatrixXd&, MatrixXd&, MatrixXd&, MatrixXd&, double, double, double, bool);
MatrixXd predict_esn(const MatrixXd&, MatrixXd&, MatrixXd&, MatrixXd&, MatrixXd&, double, VectorXd, bool);

#endif // !_QESN_H

