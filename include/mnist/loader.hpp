/*
 * loader.hpp
 *
 *  Created on: Feb 25, 2017
 *      Author: godoakos
 */

#ifndef INCLUDE_MNIST_LOADER_HPP_
#define INCLUDE_MNIST_LOADER_HPP_

#include <vector>
#include <algorithm>
#include "mnist_reader.hpp"
#include "mnist_utils.hpp"

using namespace std;

/*
 * JUST SOME QUICK WRAPPERS FOR THE MNIST LOADER
 * https://github.com/wichtounet/mnist
 * CONVERTS IMAGES TO DOUBLES,
 * LABELS TO ONE-HOT ENCODED CLASSES
 */

vector<vector<double> > load_training_imgs(){
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	mnist::normalize_dataset(dataset);
	vector<vector<double> > imgs(dataset.training_images.size(),vector<double>(dataset.training_images[0].size(),0.));

	for(int j=0;j<dataset.training_images.size();j++){
		for(int i=0;i<dataset.training_images[j].size();i++){
			imgs[j][i] = double(dataset.training_images[j][i]);
		}
	}
	return imgs;
}

vector<vector<double> > load_training_labels(){
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	vector<vector<double> > labels(dataset.training_labels.size(),vector<double>(10,0.));
	for(int i=0;i<labels.size();i++){
		labels[i][int(dataset.training_labels[i])] = 1.;
	}
	return labels;
}

vector<vector<double> > load_test_imgs(){
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	mnist::normalize_dataset(dataset);
	vector<vector<double> > imgs(dataset.test_images.size(),vector<double>(dataset.test_images[0].size(),0.));

	for(int j=0;j<dataset.test_images.size();j++){
		for(int i=0;i<dataset.test_images[j].size();i++){
			imgs[j][i] = double(dataset.test_images[j][i]);
		}
	}
	return imgs;
}

vector<vector<double> > load_test_labels(){
	auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
	vector<vector<double> > labels(dataset.test_labels.size(),vector<double>(10,0.));
	for(int i=0;i<labels.size();i++){
		labels[i][int(dataset.test_labels[i])] = 1.;
	}
	return labels;
}
#endif /* INCLUDE_MNIST_LOADER_HPP_ */
