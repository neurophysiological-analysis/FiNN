/*
 * eigen_decomp.h
 *
 *  Created on: Jul 15, 2024
 *      Author: max
 */

#ifndef EIGEN_DECOMP_H_
#define EIGEN_DECOMP_H_

#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <string>
#include <eigen3/unsupported/Eigen/MPRealSupport>

#include "npy.hpp"

extern "C" {
	void finnpy_eigen_decomp(double*, unsigned int,
							 double*, double*,
							 unsigned int);
}

#endif /* EIGEN_DECOMP_H_ */
