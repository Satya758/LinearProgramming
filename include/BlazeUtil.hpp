//
// Created by satya on 17/9/15.
//

#ifndef LINEARPROGRAMMING_BLAZEUTIL_HPP
#define LINEARPROGRAMMING_BLAZEUTIL_HPP

#include "Problem.hpp"

namespace lp {

class SplitVector {
 public:
  const DenseSubvector x;
  const DenseSubvector y;
  const DenseSubvector z;

  SplitVector(const Problem& problem, const DenseVector& vec)
      : x(blaze::subvector(vec, 0UL, problem.columns)),
        y(blaze::subvector(vec, problem.columns, problem.equalityRows)),
        z(blaze::subvector(vec, problem.columns + problem.equalityRows,
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
  return std::max({normInf(vec1), normInf(vec2), normInf(vec3)});
}
}
#endif  // LINEARPROGRAMMING_BLAZEUTIL_HPP
