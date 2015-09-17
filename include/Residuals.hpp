//
// Created by satya on 5/9/15.
//

#ifndef LINEARPROGRAMMING_RESIDUAL_HPP
#define LINEARPROGRAMMING_RESIDUAL_HPP

#include "Problem.hpp"

namespace lp {

/**
 * x, y, z are firstBlock, secondBlock and z thirdBlock respectively
 */
class Residuals {
  // To respect order if initialization, members are specified in a order
 private:
  const Problem& _problem;
  const DenseVector& _rhs;
  const DenseVector& _solution;
  const DenseVector& _omegaSquare;
  const bool _withStaticRegularization;
  const DenseSubvector _rhsX;
  const DenseSubvector _rhsY;
  const DenseSubvector _rhsZ;
  const DenseSubvector _solX;
  const DenseSubvector _solY;
  const DenseSubvector _solZ;

  /**
   * rhsX -(0 or staticDelta)*solX - A'*solY - G'*solZ
   */
  DenseVector getRx() const {
    DenseVector rX;

    rX = _rhsX - blaze::trans(_problem.A) * _solY -
         blaze::trans(_problem.G) * _solZ;

    if (_withStaticRegularization) {
      rX -= _problem.options.staticDelta * _solX;
    }

    return rX;
  }

  /**
   * rhsY - A * solX + (0 or staticDelta) * solY
   */
  DenseVector getRy() const {
    DenseVector rY;

    rY = _rhsY - _problem.A * _solX;

    if (_withStaticRegularization) {
      rY += _problem.options.staticDelta * _solY;
    }

    return rY;
  }

  /**
   * rhsZ - G * solX - (-W^2) * solZ
   * Note: omegaSquare is already negative semi definite
   */
  DenseVector getRz() const {
    DenseVector rZ;

    rZ = _rhsZ - _problem.G * _solX - _omegaSquare * _solZ;

    return rZ;
  }

  DenseSubvector getSubvector(const DenseVector& vec, const size_t startIndex,
                              const size_t size) const {
    DenseSubvector subvector = blaze::subvector(vec, startIndex, size);

    return subvector;
  }

 public:
  const DenseVector x;
  const DenseVector y;
  const DenseVector z;

  Residuals(const Problem& problem, const DenseVector& rhs,
            const DenseVector& solution, const DenseVector& omegaSquare,
            const bool withStaticRegularization = false)
      : _problem(problem),
        _rhs(rhs),
        _solution(solution),
        _omegaSquare(omegaSquare),
        _withStaticRegularization(withStaticRegularization),
        _rhsX(getSubvector(_rhs, 0UL, _problem.columns)),
        _rhsY(getSubvector(_rhs, _problem.columns, _problem.equalityRows)),
        _rhsZ(getSubvector(_rhs, _problem.columns + _problem.equalityRows,
                           _problem.inequalityRows)),
        _solX(getSubvector(_solution, 0UL, _problem.columns)),
        _solY(getSubvector(_solution, _problem.columns, _problem.equalityRows)),
        _solZ(getSubvector(_solution, _problem.columns + _problem.equalityRows,
                           _problem.inequalityRows)),
        x(getRx()),
        y(getRy()),
        z(getRz()) {}
};
}
#endif  // LINEARPROGRAMMING_RESIDUAL_HPP
