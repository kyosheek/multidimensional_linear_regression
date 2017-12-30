#include <iostream>
#include "linearregression.h"

using namespace std;

LinearRegression::LinearRegression(double x[], double y[], int m) {
	this->x = x;
	this->y = y;
	this->m = m;
}

void LinearRegression::train(double alpha, int iter) {
	auto *J = new double[iter];
	this->t = graident_descent(x, y, alpha, J, iter, m);

	cout << "J = ";
	for (int i = 0; i < iter; ++i) {
		cout << J[i] << ' ';
	}
	cout << endl;
	cout << "Theta: " << t[0] << ' ' << t[1] << endl;
}

double LinearRegression::predict(double x) {
	return h(x, t);
}

double LinearRegression::calc_cost(double x[], double y[], double t[], int m) {
	double *preds = calc_pred(x, t, m);

	auto *diff = new double[m];
	for (int i = 0; i < m; ++i) { diff[i] = preds[i] - y[i]; }

	auto *sq_errors = new double[m];
	for (int i = 0; i < m; ++i) { sq_errors[i] = pow(diff[i], 2); }

	double s = 0;
	for (int i = 0; i < m; ++i) { s += sq_errors[i]; }

	return ((1.0 / (2 * m)) * s);
}

double LinearRegression::h(double x, double t[]) {
	return t[0] + x * t[1];
}

double *LinearRegression::calc_pred(double x[], double t[], int m) {
	auto *pred = new double[m];

	//calculate h for each training data set
	for (int i = 0; i < m; ++i) { pred[i] = LinearRegression::h(x[i], t); }

	return pred;
}

double *LinearRegression::graident_descent(double x[], double y[], double alpha, double *J, int iter, int m) {
	auto t = new double[2];
	t[0] = 1; t[1] = 1;

	for (int i = 0; i < iter; ++i) {
		auto *pred = LinearRegression::calc_pred(x, t, m);
		auto *diff = new double[m];
		for (int j = 0; j < m; ++j) { diff[j] = pred[j] - y[j]; }

		auto *errors_x1 = diff;
		auto *errors_x2 = new double[m];
		for (int j = 0; j < m; ++j) { errors_x2[j] = diff[j] * x[j]; }

		double sum = 0;
		for (int j = 0; j < m; ++j) { sum += errors_x1[j]; }
		t[0] = t[0] - alpha * (1.0 / m) * sum;

		sum = 0;
		for (int j = 0; j < m; ++j) { sum += errors_x2[j]; }
		t[1] = t[1] - alpha * (1.0 / m) * sum;

		J[i] = LinearRegression::calc_cost(x, y, t, m);
	}

	return t;
}