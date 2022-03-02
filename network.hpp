/*
 * network.hpp
 *
 *  Created on: Feb 25, 2017
 *      Author: godoakos
 */

#ifndef NETWORK_HPP_
#define NETWORK_HPP_

#include <vector>
#include <algorithm>
#include "nonlinearities.hpp"

using namespace std;

class network{

	int in_size;
	int classes;

	enum nl_type {sigm = 0, rectifier = 1};

	double eta; //learning rate
	double decay = 1.; //weight decay/regularization term

	vector<vector<vector<double> > > w;
	vector<vector<vector<double> > > w_upd;
	vector<vector<double> > b_upd;
	vector<nonlinearity*> nl;
	vector<vector<double> > z;
	vector<vector<double> > b;


public:

	network(int input_size, int num_classes, vector<int> layer_sizes = {}, double learning_rate = 0.1, nonlinearity* nonlin = new sigmoid()) : in_size(input_size), classes(num_classes), eta(learning_rate){
		if (layer_sizes.empty()) {
			w.push_back(vector<vector<double> >(num_classes,vector<double> (input_size, 0.)));
			z.push_back(vector<double>(num_classes, 0.));
			b.push_back(vector<double>(num_classes, 1.));
		}
		else{
			w.push_back(vector<vector<double> >(layer_sizes[0],vector<double> (input_size, 0.)));
			z.push_back(vector<double>(layer_sizes[0], 0.));
			b.push_back(vector<double>(layer_sizes[0], 1.));
			for(int i=1;i<layer_sizes.size();i++){
				w.push_back(vector<vector<double> >(layer_sizes[i],vector<double>(layer_sizes[i-1], 0.)));
				z.push_back(vector<double>(layer_sizes[i], 0.));
				b.push_back(vector<double>(layer_sizes[i], 1.));
			}
			w.push_back(vector<vector<double> >(num_classes,vector<double>(layer_sizes[layer_sizes.size()-1])));
			z.push_back(vector<double>(num_classes, 0.));
			b.push_back(vector<double>(num_classes, 1.));
		}

		w_upd = w;
		b_upd = b;

		for(int i=0; i<w.size()-1;i++){
			nl.push_back(nonlin);
		}
		nl.push_back(new softmax());

		random_device rd;
		mt19937 gen(rd());
		normal_distribution<double> dis(0.0,1.0);

		for(int l=0;l<w.size();l++){
			for(int i=0;i<w[l].size();i++){
				for(int j=0;j<w[l][i].size();j++){
					w[l][i][j] = dis(gen);
				}
			}
		}

	}

	void set_nonlin(nonlinearity* nonlin, unsigned idx){
		/*
		 * CHECK nonlinearities.hpp FOR AVAILABLE ONES
		 */
		if (idx>nl.size()-2) return;
		delete nl[idx];
		nl[idx] = nonlin;
	}

	void set_all_nonlin(nonlinearity* nonlin){
		/*
		 * CHECK nonlinearities.hpp FOR AVAILABLE ONES
		 */
		cout << nl.size() << ' ';
		for(int i = 0;i< nl.size()-1;i++){
			delete nl[i];
			nl[i] = nonlin;
		}
		cout << nl.size() << endl;
	}

	void step_learning_rate(double step){
		eta *= step;
	}

	vector<double> forward(vector<double> &input){
		//INIT
		for(int l=0;l<z.size();l++){
			for(int i=0;i<z[l].size();i++){
				z[l][i] = 0.;
			}
		}
		vector<double> a_prev(input);
		//PASSING FORWARD
		for(int l=0;l<w.size();l++){
			for(int i=0; i<w[l].size();i++){
				for(int j=0; j<w[l][i].size();j++){
					z[l][i] += w[l][i][j] * a_prev[j];
				}
				z[l][i] += b[l][i];
			}
			a_prev = nl[l]->fwd(z[l]);
		}
		/*
		 * LITTLE CHEATING GOING ON HERE,
		 * LAST ELEMENT OF Z IS ACTUALLY THE NETWORK OUTPUT
		 */
		z[z.size()-1] = nl[nl.size()-1]->fwd(z[z.size()-1]);
		return (z[z.size()-1]);
	}

	void set_regularization_term(double lambda,unsigned dataset_size){
		decay = 1-(eta*lambda)/dataset_size;
	}

	vector<double> error(vector<double> &label){ //should be probability already
		/*
		 * RETURNS THE OUTPUT LAYER'S ERROR, NOT THE LOSS!!!
		 */
		vector<double> error;
		for(int i=0;i<label.size();i++)
			error.push_back(z[z.size()-1][i] - label[i]);
		return error;
	}


	void backprop(vector<double> &sample , vector<double> &label){

		/*
		 * ACCUMULATES GRADIENT FOR A SAMPLE + LABEL
		 * MAKE SURE THAT UPDATES ARE INITED TO 0 BEFORE THIS!!!
		 */
		vector<double> delta_next = error(label);
		vector<double> a_prev;

		//LAST LAYER
		if (nl.size()<2){ //not much to do if there are no hidden layers
			a_prev = sample;
			for(int j=0;j<w[w.size()-1][0].size();j++){
				for(int i=0;i<w[w.size()-1].size();i++){
					w_upd[w.size()-1][i][j] += a_prev[j] * delta_next[i];//a little cheat, didn't really want to move the last layer's error around too much
				}
			}
			return;
		}

		vector<double> z_curr;
		vector<double> delta_curr;
		a_prev = nl[nl.size()-2]->fwd(z[z.size()-2]);

		for(int j=0;j<w[w.size()-1][0].size();j++){
			for(int i=0;i<w[w.size()-1].size();i++){
				w_upd[w.size()-1][i][j] += a_prev[j] * delta_next[i];//a little cheat, didn't really want to move the last layer's error around too much
			}
		}

		//HIDDEN LAYERS, EXCEPT FIRST l=L-1...1
		for(int l=w.size()-2;l>0;l--){

			delta_curr = vector<double>(b[l].size(),0.); // d_l - exactly as many neuron errors as neuron biases
			z_curr = nl[l] -> bwd(z[l]); // z_l
			a_prev = nl[l-1] -> fwd(z[l-1]); //a_l-1

			//CURRENT ERROR
			for(int j=0;j<w[l+1][0].size();j++){
				for(int i=0;i<w[l+1].size();i++){
					delta_curr[j] += w[l+1][i][j] * delta_next[i];
				}
				delta_curr[j] *= z_curr[j];
			}

			//ACCUMULATE UPDATES
			for(int i=0; i<w_upd[l].size();i++){
				b_upd[l][i] = delta_curr[i];
				for(int j=0; j<w_upd[l][i].size();j++){
					w_upd[l][i][j] += a_prev[j] * delta_curr[i];
				}
			}
			//STEPPING BACK WITH ERROR
			delta_next = delta_curr;
		}

		//FIRST LAYER, l=0

		delta_curr = vector<double>(b[0].size(),0.); // d_l - exactly as many neuron errors as neuron biases
		z_curr = nl[0] -> bwd(z[0]); // z_l
		a_prev = sample; // the input layer is a virtual -1st layer

		//CURRENT ERROR
		for(int j=0;j<w[1][0].size();j++){
			for(int i=0;i<w[1].size();i++){
				delta_curr[j] += w[1][i][j] * delta_next[i];
			}
			delta_curr[j] *= z_curr[j];
		}
		//ACCUMULATE UPDATES
		for(int i=0; i<w_upd[0].size();i++){
			b_upd[0][i] = delta_curr[i];
			for(int j=0; j<w_upd[0][i].size();j++){
				w_upd[0][i][j] += a_prev[j] * delta_curr[i];
			}
		}
	}

	void train(vector<vector<double> > &samples, vector<vector<double> > &labels){
		//CLEAR UPDATES BEFORE BACKPROP
		for(int l=0;l<w_upd.size();l++){
			for(int i=0;i<w_upd[l].size();i++){
				b_upd[l][i] = 0.;
				for(int j=0;j<w_upd[l][i].size();j++){
					w_upd[l][i][j] = 0.;
				}
			}
		}

		//ACCUMULATE GRADIENT
		for(int i=0;i<samples.size();i++){
			forward(samples[i]);
			backprop(samples[i], labels[i]);
		}



		//DESCENT
		for(int l=0;l<w_upd.size();l++){
			for(int i=0;i<w_upd[l].size();i++){
				b[l][i] = decay*b[l][i]-(b_upd[l][i])*(eta/double(samples.size()));
				for(int j=0;j<w_upd[l][i].size();j++){
					w[l][i][j] = decay*w[l][i][j] - w_upd[l][i][j]*(eta/double(samples.size()));
				}
			}
		}
	}

	unsigned classify(vector<double> &sample){
		vector<double> output = forward(sample);
		return distance(output.begin(),max_element(output.begin(),output.end()));
	}

	~network(){
		for(int i=0;i<nl.size();i++){
			delete nl[i];
		}
	}

};



#endif /* NETWORK_HPP_ */
