//============================================================================
// Name        : mlp_mini.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "network.hpp"
#include "include/mnist/loader.hpp"
using namespace std;

void test(network &net){
	cout << "testing... " << endl;
	vector<vector<double> > samples = load_test_imgs();
	vector<vector<double> > labels = load_test_labels();
	unsigned hit = 0;
	for(int i=0;i<samples.size();i++){
		hit += (net.classify(samples[i]) == distance(labels[i].begin(), max_element(labels[i].begin(), labels[i].end())));
	}
	cout << "test results (validation data): " << hit << " / " << samples.size() << " -  " << 100.*(double(hit)/double(samples.size()))<< "%" <<  endl;
}

void test_on_training_samples(network& net){
	cout << "testing... " << endl;
	vector<vector<double> > samples = load_training_imgs();
	vector<vector<double> > labels = load_training_labels();
	unsigned hit = 0;
	for(int i=0;i<samples.size();i++){
		hit += (net.classify(samples[i]) == distance(labels[i].begin(), max_element(labels[i].begin(), labels[i].end())));
	}
	cout << "test results (training data): " << hit << " / " << samples.size() << " -  " << 100.*(double(hit)/double(samples.size()))<< "%" <<  endl;
}


void train(network &net, unsigned batch_size = 10, unsigned num_epochs = 10, int step_interval = -1, double step_magnitude = 1.){
	/*
	 * LOADS DATA AND HANDLES BATCHES
	 * CAN MODIFY LEARNING RATE AS IT GOES
	 */
	vector<vector<double> > samples = load_training_imgs();
	vector<vector<double> > labels = load_training_labels();
	vector<vector<double>> batch_samples;
	vector<vector<double>> batch_labels;

	net.set_regularization_term(0.1, samples.size());
	for(int i=0;i<num_epochs;i++){
		cout << "epoch " << i+1 << " / " << num_epochs << endl;
		if(step_interval>0 && i%step_interval==0 && i>0) net.step_learning_rate(step_magnitude);
		unsigned idx = 0;
		while(idx+batch_size<samples.size()){ //do as many batches as possible
			if(idx%1000==0 && idx>0) cout << idx << " / " << samples.size() << endl;

			batch_samples = vector<vector<double> >(samples.begin()+idx, samples.begin()+idx+batch_size);
			batch_labels = vector<vector<double> >(labels.begin()+idx, labels.begin()+idx+batch_size);

			net.train(batch_samples, batch_labels);

			idx += batch_size;
		}
		if(idx<samples.size()){ //do the rest
			batch_samples = vector<vector<double> >(samples.begin()+idx, samples.end());
				batch_labels = vector<vector<double> >(labels.begin()+idx, labels.end());
				net.train(batch_samples, batch_labels);
		}
		test_on_training_samples(net);
		test(net);
	}
}


int main() {
	unsigned NUM_EPOCHS = 100;
	unsigned BATCH_SIZE = 100;

	/*
	 * REFER TO SECTION C5 IN
	 * http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
	 */

	//network c5_1(784, 10, {1000}, 0.1);
	network c6_1(784, 10, {1000,150}, 0.1);

	//train(c5_1, BATCH_SIZE, NUM_EPOCHS, 30, 0.1);
	train(c6_1, BATCH_SIZE, NUM_EPOCHS, 30, 0.1);

	/*
	network nohidden(784,10,{}, 0.1);
	train(nohidden, BATCH_SIZE, NUM_EPOCHS);//*/

	return 0;
}
