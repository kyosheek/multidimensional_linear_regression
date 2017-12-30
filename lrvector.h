#pragma once
#ifndef LRVECTOR_H
#define LRVECTOR_H

#include <vector>

using namespace std;

class LRMatrix {
public:
	vector<vector<double>> x;
	vector<double> y;
	int m;
	vector<double> t;

	LRMatrix(vector<vector<double>> x, vector<double> y, int m);

	void train(double alpha, int iter);
	double predict(vector<double> x);
	
private:
	
	static double calc_cost(vector<vector<double>> x, vector<double> y, vector<double> t, int m);
	static double h(vector<double> x, vector<double> t);
	static vector<double> calc_pred(vector<vector<double>> x, vector<double> t, int m);
	static vector<double> graident_descent(vector<vector<double>> x, vector<double> y, double alpha, double *J, int iter, int m);
};

#endif //LRVECTOR_H
