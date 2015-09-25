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
    // TODO Do we need epsilon? Does z value can go so small!
    for (size_t j = 0; j < problem.inequalityRows; ++j) {
      const double sValue = point.s[j];
      const double zValue = point.z[j];

      // Its not -W^2 but W^2
      omegaSquare[j] =
          (sValue / (zValue < problem.options.epsilon ? problem.options.epsilon
                                                      : zValue));
      omega[j] = std::sqrt(std::abs(omegaSquare[j]));

      lambdaSquare[j] = sValue * zValue;
      lambda = std::sqrt(lambdaSquare[j]);
    }
  }

  /**
   * Intial scalings, lambda, lambdaSqaure are not used
   */
  NTScalings(const Problem& problem)
      : omega(problem.inequalityRows, 1),
        omegaSquare(problem.inequalityRows, 1),
        lambda(0),
        lambdaSquare(0) {}

  DenseVector omega;
  DenseVector omegaSquare;

  DenseVector lambda;
  DenseVector lambdaSquare;
};

}  // lp

#endif  // NTSCALINGS_HPP
