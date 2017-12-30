#pragma once
#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

class LinearRegression {
public:
	//first values
	double *x;

	//target values
	double *y;

	//number of (training) examples
	int m;

	//theta coefficients
	double *t;

	LinearRegression(double x[], double y[], int m);

	//void train
	//performs iter gradient steps with alpha coefficient
	void train(double alpha, int iter);

	//void predict
	//try to predict y with given x
	double predict(double x);

private:

	//double calc_cost
	//calculates the cost J with given data
	static double calc_cost(double x[], double y[], double t[], int m); // J(t) = 1/2m * sum(m) [ h(t)(x(m) - y(m))^2 ], m = 0 ... n

	//double h
	//calculates the hypothesis
	static double h(double x, double t[]); //h = t[0] + t[1] * x + ... + t[n] * x ^ (n-1)

	//double *calc_pred
	//calculates target value with given other values
	static double *calc_pred(double x[], double t[], int m);

	//double *gradient_descend
	//performs gradient descent to learn theta by taking iters gradient steps with learn ratio alpha
	static double *graident_descent(double x[], double y[], double alpha, double *J, int iter, int m);
};

#endif //LINEARREGRESSION_H