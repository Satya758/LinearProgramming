//
// Created by satya on 5/9/15.
//

#ifndef POINT_HPP
#define POINT_HPP

#include "Problem.hpp"

namespace lp {
/**
 * Primal Dual point.
 */
class Point {
 public:
  Point(const Problem& problem)
      : x(problem.columns),
        s(problem.inequalityRows),
        y(problem.equalityRows),
        z(problem.inequalityRows) {}

  // Primal variables
  DenseVector x;
  DenseVector s;

  // Dual variables
  DenseVector y;
  DenseVector z;

  // Homogenizing variables
  double tau;
  double kappa;
};

}  // lp
#endif  // POINT_HPP
