/*
 * nonlinearities.hpp
 *
 *  Created on: Feb 23, 2017
 *      Author: godoakos
 */

#ifndef INCLUDE_NONLINEARITIES_HPP_
#define INCLUDE_NONLINEARITIES_HPP_

#include <math.h>
#include <iostream>
#include <algorithm>

using namespace std;

class nonlinearity{
public:
	virtual vector<double> bwd(vector<double> x) = 0;
	virtual vector<double> fwd(vector<double> x) = 0;
	virtual ~nonlinearity(){}
};

class sigmoid : public nonlinearity{
	/*
	 * CLASSIC SIGMOID NONLINEARITY
	 */
public:
	vector<double> bwd(vector<double> x){
		vector<double> v;
		for(double x_i : x)
			v.push_back((1.0/(1.0 + exp(-x_i))) * (1.0 - (1.0/(1.0 + exp(-x_i)))));
		return v;
	}
	vector<double> fwd(vector<double> x){
		vector<double> v;
		for(double x_i : x)
			v.push_back((1.0/(1.0 + exp(-x_i))));
		return v;
	}
};

class relu : public nonlinearity{
	/*
	 * (LEAKY) RELU NONLINEARITY
	 */
	double epsilon;

public:
	relu() : epsilon(0.) {} //plain relu
	relu(double param) : epsilon(param) {} //leaky relu

	vector<double> bwd(vector<double> x){
		vector<double> v;
		for(int i=0; i<x.size();i++){
			v.push_back((x[i]>=0) ? 1. : epsilon); //slope of curve
		}
		return v;
	}

	vector<double> fwd(vector<double> x){
		vector<double> v;
		for(int i=0; i<x.size();i++){
			v.push_back((x[i]>=0) ? x[i] : epsilon*x[i]); // max(epsilon*x, x)
		}
		return v;
	}
};

class tan_h : public nonlinearity {

	/*
	 * HYPERBOLIC TANGENT NONLINEARITY
	 */

public:
	vector<double> bwd(vector<double> x){
		vector<double> v;
		for(int i=0; i<x.size();i++){
			v.push_back(1 - pow((2. / (1.+exp(-2. * x[i])))-1, 2));
		}
		return v;
	}

	vector<double> fwd(vector<double> x){
		vector<double> v;
		for(int i=0; i<x.size();i++){
			v.push_back((2. / (1.+exp(-2. * x[i])))-1);
		}
		return v;
	}

};

class softmax : public nonlinearity{
	/*
	 * SOFTMAX NONLIN USED IN THE OUTPUT LAYER
	 */
public:
	vector<double> fwd(vector<double> x){
		vector<double> output;
		double denom = 0.;
		double sum = 0.;
		for(int i=0; i<x.size(); i++){
			output.push_back(exp(x[i]));
			denom += exp(x[i]);
		}
		for(int i=0; i<output.size(); i++){
			output[i]/=denom;
		}
		return output;
	}

	vector<double> bwd(vector<double> x){
		return x; //not used
	}
};

#endif /* INCLUDE_NONLINEARITIES_HPP_ */
