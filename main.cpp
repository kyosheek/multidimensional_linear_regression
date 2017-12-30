#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include "linearregression.h"
#include "lrvector.h"

using namespace std;

namespace path {
	string one = "data1.txt";
	string two = "data2.txt";
}

void linearregression();
void multidimlrrest();
void multidimlrflat();

int main() {
	int num = 0;
	cout << "Enter number for task: " << endl
		<< "1 - linear regression (restaraunt's income)" << endl
		<< "2 - multidimensional linear regression (restaraunt's income)" << endl
		<< "3 - multidim lr (flat price)" << endl;
	while (true) {
		cin >> num;
		if (num == 1 || num == 2 || num == 3) { break; }
		else { cout << "Wrong task number, try again:" << endl; cin >> num; }
	}

	switch (num) {
	case 1:
		linearregression();
		break;
	case 2:
		multidimlrrest();
		break;
	case 3:
		multidimlrflat();
		break;
	default:
		break;
	}

	return 0;
}

void linearregression() {
	ifstream file;
	file.open(path::one);
	string str, res;
	int k = 1;
	vector<double> xget, yget;

	cin.ignore();
	while (file.good()) {
		if (k % 2 == 1) {
			getline(file, str, ','); 
			xget.push_back(stod(str)); 
		}
		if (k % 2 == 0) { 
			getline(file, str);
			yget.push_back(stod(str)); }
		k++;
		if (file.peek() == EOF) { break; }
	}
	k--;
	file.close();

	auto *x = new double[k/2]; auto *y = new double[k/2];
	for (int i = 0; i < k/2; ++i) {
		x[i] = xget[i];
		y[i] = yget[i];
	}
	xget.clear(); yget.clear(); str.clear();

	LinearRegression lr(x, y, k/2);

	cout << "Enter learning rate alpha: ";
	double alpha;
	cin >> alpha;

	cout << "Enter number of iterations: ";
	int iter;
	cin >> iter;

	cout << "Training model..." << endl;
	lr.train(alpha, iter);

	cout << "Model has been trained.\nNumber of people in town is 35.000 : ";
	double profit1 = lr.predict(3.5);
	cout << "Estimated income is "  << profit1 << endl;

	cout << "Number of people in town is 70.000 : ";
	double profit2 = lr.predict(7.0);
	cout << "Estimated income is " << profit2 << endl;
}

void multidimlrrest() {
	ifstream file;
	file.open(path::one);
	string str;
	int k = 1;
	vector<double> xget, y;
	vector<vector<double>> x;

	cin.ignore();
	while (file.good()) {
		if (k % 2 == 1) {
			getline(file, str, ',');
			xget.push_back(stod(str));
		}
		if (k % 2 == 0) {
			getline(file, str);
			y.push_back(stod(str));
		}
		k++;
		if (file.peek() == EOF) { break; }
	}
	k--;
	file.close();

	for (int i = 0; i < xget.size(); ++i) {
		vector<double> push = { xget[i] };
		x.push_back(push);
	}
	xget.clear(); str.clear();

	LRVector lr(x, y, k / 2);

	cout << "Enter learning rate alpha: ";
	double alpha;
	cin >> alpha;

	cout << "Enter number of iterations: ";
	int iter;
	cin >> iter;

	cout << "Training model..." << endl;
	lr.train(alpha, iter);

	cout << "Model has been trained.\nNumber of people in town is 35.000";
	vector<double> pred1 = { 3.5 };
	double profit1 = lr.predict(pred1);
	cout << "Estimated income is " << profit1 << endl;

	cout << "Number of people in town is 70.000";
	vector<double> pred2 = { 7.0 };
	double profit2 = lr.predict(pred2);
	cout << "Estimated income is " << profit2 << endl;
}

void multidimlrflat() {
	ifstream file;
	file.open(path::two);
	string str, res;
	int k = 0;
	vector<vector<double>> x;
	vector<double> xget, y;

	cin.ignore();
	while (file.good()) {
		getline(file, str, ',');
		xget.push_back(stod(str));

		getline(file, str, ',');
		xget.push_back(stod(str));

		getline(file, str);
		y.push_back(stod(str));

		x.push_back(xget);
		xget.clear();
		if (file.peek() == EOF) { break; }
		k++;
	}
	file.close();
	str.clear();

	LRVector lr(x, y, k);

	cout << "Enter learning rate alpha: ";
	double alpha;
	cin >> alpha;

	cout << "Enter number of iterations: ";
	int iter;
	cin >> iter;

	cout << "Training model..." << endl;
	lr.train(alpha, iter);

	cout << "Model has been trained.\nFlat has square of 1650 and 3 room(s) : ";
	double profit1 = lr.predict({ 1650, 3 });
	cout << "Estimated price is " << profit1 << endl;

	cout << "Flat has square of 1650 and 1 room(s) :";
	double profit2 = lr.predict({ 1650, 1 });
	cout << "Estimated income is " << profit2 << endl;
}
