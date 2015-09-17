//
// Created by satya on 17/9/15.
//

#ifndef LINEARPROGRAMMING_KKTUTIL_HPP
#define LINEARPROGRAMMING_KKTUTIL_HPP

#include "Problem.hpp"

namespace lp {

class KKTUtil {
 public:
  KKTUtil(const Problem& problem)
      : size(problem.columns + problem.equalityRows + problem.inequalityRows),
        nnz(2 * (problem.A.nonZeros() + problem.G.nonZeros()) + size),
        utNnz(problem.A.nonZeros() + problem.G.nonZeros() + size) {}

  const size_t size;
  // Number of non zeros in KKT symmetric matrix, only lower or upper part is
  // used for count (which includes diagonal elements which are non-zero)
  // kktMatrixSize is used as number of diagonal elements
  // Blaze requires non-zeros of whole matrix rather than just lower/upper
  // matrix, so there is multiplication factor of 2
  // Non zeros of complete KKT matrix
  const size_t nnz;
  // KKT non zeros in upper triangle, as only upper triangle is enough to
  // compute factorization
  const size_t utNnz;
};
}

#endif  // LINEARPROGRAMMING_KKTUTIL_HPP
