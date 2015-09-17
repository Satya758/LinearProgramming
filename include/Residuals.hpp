//
// Created by satya on 5/9/15.
//

#ifndef LINEARPROGRAMMING_RESIDUAL_HPP
#define LINEARPROGRAMMING_RESIDUAL_HPP

#include "Problem.hpp"

namespace lp {

/**
 * Though called Rx, Ry, Rz these does not represent point but first to third
 * blocks
 */
class Residuals {
 private:
  const Problem& _problem;
  const Point& _point;

  const DenseVector _subRx;
  const DenseVector _subRy;
  const DenseVector _subRz;

  const double _cTx;
  const double _bTy;
  const double _hTz;

  const double _rxNorm;
  const double _ryNorm;
  const double _rzNorm;

  /**
   * -A'*x - G'*z
   */
  DenseVector getSubRx() const {
    return -blaze::trans(_problem.A) * _point.y -
           blaze::trans(_problem.G) * _point.z;
  }

  /**
   * A*x
   */
  DenseVector getSubRy() const { return _problem.A * _point.x; }

  /**
   * G*x + s
   */
  DenseVector getSubRz() const { return _problem.G * _point.x + _point.s; }

  double getRTau() const { return _point.kappa + _cTx + _bTy + _hTz; }

  double getFeasableValue() {
    double v1 = blaze::length(-_subRx + _problem.c) / _rxNorm;
    double v2 = blaze::length(_subRy - _problem.b) / _ryNorm;
    double v3 = blaze::length(_subRz - _problem.h) / _rzNorm;

    return std::max(std::max(v1, v2), v3);
  }

  double getAbsoluteValue() { return blaze::trans(_point.s) * _point.z; }

  double getRelativeValue() {
    double denominator = std::max(std::max(-_cTx, -_bTy), -_hTz);

    return absolute / denominator;
  }

 public:
  // -A'*x - G'*z - c*tau
  const DenseVector rx;
  // A*x - b*tau
  const DenseVector ry;
  // G*x + s - h*tau
  const DenseVector rz;
  // kappa + c'*x + b'*y + h'*z
  const double rTau;

  const double feasable;
  const double absolute;
  const double relative;

  Residuals(const Problem& problem, const Point& point, const double rxNorm,
            const double ryNorm, const double rzNorm)
      : _problem(problem),
        _point(point),
        _subRx(getSubRx()),
        _subRy(getSubRy()),
        _subRz(getSubRz()),
        _cTx(blaze::trans(_problem.c) * _point.x),
        _bTy(blaze::trans(_problem.b) * _point.y),
        _hTz(blaze::trans(_problem.h) * _point.z),
        _rxNorm(rxNorm),
        _ryNorm(ryNorm),
        _rzNorm(rzNorm),
        rx(_subRx - _problem.c * _point.tau),
        ry(_subRy - _problem.b * _point.tau),
        rz(_subRz - _problem.h * _point.tau),
        rTau(getRTau()),
        feasable(getFeasableValue()),
        absolute(getAbsoluteValue()),
        relative(getRelativeValue()) {}
};
/**
 * x, y, z are firstBlock, secondBlock and z thirdBlock respectively
 * Computes KKT residuals
 */
class ResidualsKkt {
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
  DenseVector getKktRx() const {
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
  DenseVector getKktRy() const {
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
  DenseVector getKktRz() const {
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
  const DenseVector kktX;
  const DenseVector kktY;
  const DenseVector kktZ;

  // Used to compute KKT residuals
  ResidualsKkt(const Problem& problem, const DenseVector& rhs,
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
        kktX(getKktRx()),
        kktY(getKktRy()),
        kktZ(getKktRz()) {}
};
}
#endif  // LINEARPROGRAMMING_RESIDUAL_HPP
