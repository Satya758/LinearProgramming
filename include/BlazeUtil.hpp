//
// Created by satya on 17/9/15.
//

#ifndef LINEARPROGRAMMING_BLAZEUTIL_HPP
#define LINEARPROGRAMMING_BLAZEUTIL_HPP

#include "Problem.hpp"

namespace lp {

class SplitVector {
 public:
  const DenseSubvector dx;
  const DenseSubvector dy;
  const DenseSubvector dz;

  SplitVector(const Problem& problem, const DenseVector& vec)
      : dx(blaze::subvector(vec, 0UL, problem.columns)),
        dy(blaze::subvector(vec, problem.columns, problem.equalityRows)),
        dz(blaze::subvector(vec, problem.columns + problem.equalityRows,
                            problem.inequalityRows)) {}
};

double normInf(const DenseVector& bVec) {
  double norm = 0;

  for (DenseVector::ConstIterator iter = blaze::cbegin(bVec);
       iter != blaze::cend(bVec); ++iter) {
    norm = std::max(norm, std::abs(*iter));
  }

  return norm;
}

/**
 * Infinity norm of combined three vectors
 *
 */
double normInf(const DenseVector& vec1, const DenseVector& vec2,
               const DenseVector& vec3) {
  return std::max(std::max(normInf(vec1), normInf(vec2)), normInf(vec3));
}
}
#endif  // LINEARPROGRAMMING_BLAZEUTIL_HPP
