/*
 * eigen_decomp.cpp
 *
 *  Created on: Jul 15, 2024
 *      Author: max
 */

/*
 * eigen_decomp.cpp
 *
 *  Created on: Jul 14, 2024
 *      Author: voodoocode
 */

#include "include/finnpy_speedups.h"

#include <chrono>

extern "C" {

	void finnpy_eigen_decomp(double* data, unsigned int size,
							 double* evals, double* evecs,
							 unsigned int precision = 256) {
		mpfr::mpreal::set_default_prec(precision);
		typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> MatrixXmp;
		Eigen::SelfAdjointEigenSolver<MatrixXmp> es;

		MatrixXmp loc_data = MatrixXmp::Random(size, size);
		for (unsigned int i = 0; i < size; ++i) {
			for (unsigned int j = 0; j < size; ++j) {
				//loc_data(i, j) = d.data[int(j + d.shape[1] * i)];
				loc_data(i, j) = data[int(j + size * i)];
			}
		}

		es.compute(loc_data);

		for (unsigned int i = 0; i < size; ++i) {
			for (unsigned int j = 0; j < size; ++j) {
				evecs[int(j + size * i)] = double(es.eigenvectors()(int(j + size * i)));
			}
			evals[i] = double(es.eigenvalues()(i));
		}

		return;
	}

}

/*
int main() {
	std::chrono::time_point start2 = std::chrono::high_resolution_clock::now();
	mpfr::mpreal::set_default_prec(256);
	typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> MatrixXmp;
	Eigen::SelfAdjointEigenSolver<MatrixXmp> es;

	const std::string path {"/home/max/Downloads/data.npy"};
	npy::npy_data d = npy::read_npy<double>(path);
	MatrixXmp loc_data = MatrixXmp::Random(d.shape[0], d.shape[1]);
	for (unsigned int i = 0; i < d.shape[0]; ++i) {
		for (unsigned int j = 0; j < d.shape[1]; ++j) {
			loc_data(i, j) = d.data[int(j + d.shape[1] * i)];
		}
	}
	es.compute(loc_data);

	std::cout << es.eigenvalues() << std::endl;

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_d;
	Eigen::MatrixXd loc_data_d; loc_data_d.resize(d.shape[0], d.shape[1]);
	for (unsigned int i = 0; i < d.shape[0]; ++i) {
		for (unsigned int j = 0; j < d.shape[1]; ++j) {
			loc_data_d(i, j) = d.data[int(j + d.shape[1] * i)];
		}
	}
	es_d.compute(loc_data_d);


	float error = 0;
	for (unsigned int i = 0; i < es_d.eigenvectors().size(); ++i) {
		error += double(es.eigenvectors()(i)) - double(es_d.eigenvectors()(i));
	}
	std::cout << error << std::endl;

	std::chrono::time_point start = std::chrono::high_resolution_clock::now();
	es.compute(loc_data);
	std::chrono::time_point end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
	//std::cout << es.eigenvalues() << std::endl;


	std::chrono::time_point end2 = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << std::endl;

	return 0;
}
*/
