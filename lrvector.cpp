#include <iostream>
#include "lrvector.h"

using namespace std;

LRVector::LRVector(vector<vector<double>> x, vector<double> y, int m) {
	this->x = x;
	this->y = y;
	this->m = m;
}

void LRVector::train(double alpha, int iter) {
	auto *J = new double[iter];
	this->t = graident_descent(x, y, alpha, J, iter, m);

	cout << "J = ";
	for (int i = 0; i < iter; ++i) {
		cout << J[i] << ' ';
	}
	cout << endl;

	cout << "Theta: ";
	for (auto it = t.begin(); it != t.end(); ++it) {
		cout << *it << ' ';
	}
}

double LRVector::predict(vector<double> x) {
	return h(x, t);
}

double LRVector::calc_cost(vector<vector<double>> x, vector<double> y, vector<double> t, int m) {
	vector<double> preds = calc_pred(x, t, m);

	vector<double> diff;
	for (int i = 0; i < m; ++i) { diff.push_back(preds[i] - y[i]); }

	vector<double> sq_errors;
	for (int i = 0; i < m; ++i) { sq_errors.push_back(pow(diff[i], 2)); }

	double s = 0;
	for (int i = 0; i < m; ++i) { s += sq_errors[i]; }

	return ((1.0 / (2 * m)) * s);
}

double LRVector::h(vector<double> x, vector<double> t) {
	double hyp = 0;
	for (int i = 0; i < t.size(); ++i) {
		if (i == 0) { hyp += t[i]; }
		if (i > 0) { hyp += t[i] * x[i - 1]; }
	}
	return hyp;
}

vector<double> LRVector::calc_pred(vector<vector<double>> x, vector<double> t, int m) {
	vector<double> pred;

	//vector size
	//calculate h for each training data set
	for (int i = 0; i < m; ++i) { pred.push_back(LRVector::h(x[i], t)); }

	return pred;
}

vector<double> LRVector::graident_descent(vector<vector<double>>x, vector<double> y, double alpha, double *J, int iter, int m) {
	vector<double> t;
	for (int i = 0; i <= x[0].size(); i++) { t.push_back(1); }

	for (int i = 0; i < iter; ++i) {
		vector<double> pred = LRVector::calc_pred(x, t, m);
		vector<double> diff;
		for (int j = 0; j < m; ++j) { diff.push_back(pred[j] - y[j]); }
		
		vector<vector<double>> errors;
		errors.push_back(diff);

		vector<double> toadd;
		for (auto it = 0; it != x[0].size(); ++it) {
			for (int j = 0; j < m; ++j) {
				toadd.push_back(diff[j] * x[j].at(it));
			}
			errors.push_back(toadd);
			toadd.clear();
		}

		double sum = 0;
		for (int j = 0; j < t.size(); ++j) {
			for (auto it = errors[j].begin(); it != errors[j].end(); ++it) {
				sum += *it;
			}
			t[j] = t[j] - alpha * (1.0 / m) * sum;
			sum = 0;
		}

		J[i] = LRVector::calc_cost(x, y, t, m);
	}

	return t;
}
