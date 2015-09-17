//
// Created by satya on 13/9/15.
//
#ifndef NTSCALINGS_HPP
#define NTSCALINGS_HPP

#include "Problem.hpp"
#include "Point.hpp"

namespace lp {

/**
 * Nesterov Todd scaling, Only positive orthant is considered for now
 *
 * omega  = sqrt(s/z)
 * lambda = sqrt(s.z)
 * lambdaSquare = lambda * lambda
 *
 */
class NTScalings {
 public:
  /**
   * Members of this class can be const but if they were const we cannot
   * initialize all of them in single for loop, we need separate methods to
   * initialize each of them which is unnecessary so I have removed constant
   * and initializing all of them in a single for loop as shown below, catch is
   * user using this class should create it as const
   *
   * FIXME Looks like there are no bound checks on DenseVector be careful
   */
  NTScalings(const Problem& problem, const Point& point)
      : omega(problem.inequalityRows),
        omegaSquare(problem.inequalityRows),
        lambda(problem.inequalityRows),
        lambdaSquare(problem.inequalityRows) {
    // FIXME Move to options if its used elsewhere
    const double epsilon = 1e-13;

    // TODO Do we need epsilon? Does z value can go so small!
    for (size_t j = 0; j < problem.inequalityRows; ++j) {
      const double sValue = point.s[j];
      const double zValue = point.z[j];

      // Notice negative and abs function to compute omegaSquare and omega
      // Though omegaSquare is not actually negative semi definite, we need it
      // as NSD in all our calculations with it
      omegaSquare[j] = -(sValue / (zValue < epsilon ? epsilon : zValue));
      omega[j] = std::sqrt(std::abs(omegaSquare[j]));

      lambdaSquare[j] = sValue * zValue;
      lambda = std::sqrt(lambdaSquare[j]);
    }
  }

  DenseVector omega;
  DenseVector omegaSquare;

  DenseVector lambda;
  DenseVector lambdaSquare;
};

}  // lp

#endif  // NTSCALINGS_HPP
