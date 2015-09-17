//
// Created by satya on 5/9/15.
//

#ifndef LINEARPROGRAMMING_RESIDUAL_HPP
#define LINEARPROGRAMMING_RESIDUAL_HPP

#include "Problem.hpp"
#include "BlazeUtil.hpp"

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
  // sv means subvector
  const SplitVector _svRhs;
  const SplitVector _svSol;

  /**
   * rhsX -(0 or staticDelta)*solX - A'*solY - G'*solZ
   */
  DenseVector getKktRx() const {
    DenseVector rX;

    rX = _svRhs.x - blaze::trans(_problem.A) * _svSol.y -
         blaze::trans(_problem.G) * _svSol.z;

    if (_withStaticRegularization) {
      rX -= _problem.options.staticDelta * _svSol.x;
    }

    return rX;
  }

  /**
   * rhsY - A * solX + (0 or staticDelta) * solY
   */
  DenseVector getKktRy() const {
    DenseVector rY;

    rY = _svRhs.y - _problem.A * _svSol.x;

    if (_withStaticRegularization) {
      rY += _problem.options.staticDelta * _svSol.y;
    }

    return rY;
  }

  /**
   * rhsZ - G * solX - (-W^2) * solZ
   * Note: omegaSquare is already negative semi definite
   */
  DenseVector getKktRz() const {
    DenseVector rZ;

    rZ = _svRhs.z - _problem.G * _svSol.x - _omegaSquare * _svSol.z;

    return rZ;
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
        _svRhs(_problem, _rhs),
        _svSol(_problem, _solution),
        kktX(getKktRx()),
        kktY(getKktRy()),
        kktZ(getKktRz()) {}
};
}
#endif  // LINEARPROGRAMMING_RESIDUAL_HPP
