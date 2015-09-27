//
// Created by satya on 5/9/15.
//

#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "spdlog/spdlog.h"

#include "Problem.hpp"
#include "Point.hpp"
#include "NTScalings.hpp"
#include "KKTUtil.hpp"
#include "Residuals.hpp"
#include "BlazeUtil.hpp"

namespace lp {

template <typename LinearSolver>
class Solver {
 public:
  Solver(const Problem& problem)
      : _problem(problem),
        _kktUtil(KKTUtil(_problem)),
        _lSolver(problem, _kktUtil),
        _logger(spdlog::stdout_logger_mt("Solver")) {}

  void solve() {
    _logger->info("Solver started");

    const double rxNorm = std::max(1.0, blaze::length(_problem.c));
    const double ryNorm = std::max(1.0, blaze::length(_problem.b));
    const double rzNorm = std::max(1.0, blaze::length(_problem.h));

    Point currentPoint = getInitialPoint();

    for (int j = 0; j < _problem.options.maximumIterations; ++j) {
      const Residuals residuals(_problem, currentPoint, rxNorm, ryNorm, rzNorm);
      // Check for termination conditions

      const NTScalings scalings(_problem, currentPoint);

      _lSolver.factorizeMatrix(scalings);

      DenseVector ds1 = findSolutionForRhs(scalings.omegaSquare, -_problem.c,
                                           _problem.b, _problem.h);

      const SplitVector splitDs1(_problem, ds1);

      // Common to compute both affine and combine direction, computed in
      // affineDirection
      const double tauDenominator = currentPoint.kappa / currentPoint.tau -
                                    blaze::trans(_problem.c) * splitDs1.x -
                                    blaze::trans(_problem.b) * splitDs1.y -
                                    blaze::trans(_problem.h) * splitDs1.z;

      Point affinePoint = getAffineDirection(currentPoint, residuals, scalings,
                                             splitDs1, tauDenominator);

      Point combinedPoint =
          getCombinedDirection(currentPoint, affinePoint, residuals, scalings,
                               splitDs1, tauDenominator);

      double alpha =
          computeAlpha(currentPoint, combinedPoint) * _problem.options.gamma;

      // Update point
      currentPoint.x = currentPoint.x + alpha * combinedPoint.x;
      currentPoint.y = currentPoint.y + alpha * combinedPoint.y;
      currentPoint.z = currentPoint.z + alpha * combinedPoint.z;
      currentPoint.s = currentPoint.s + alpha * combinedPoint.s;
      currentPoint.tau = currentPoint.tau + alpha * combinedPoint.tau;
      currentPoint.kappa = currentPoint.kappa + alpha * combinedPoint.kappa;
    }

    _logger->info("Solver ended");
  }

 private:
  const Problem& _problem;
  const KKTUtil _kktUtil;
  LinearSolver _lSolver;
  const std::shared_ptr<spdlog::logger> _logger;

  /**
   * Computes initial point from initial KKT matrix
   * [d   A'  G']
   * [A  -d   0 ]
   * [G   0  -I ]
   */
  Point getInitialPoint() {
    const NTScalings initialScalings(_problem);
    // Factorize Quasi PSD matrix
    _lSolver.factorizeInitialMatrix(initialScalings);
    // Create empty primal dual point
    Point point(_problem);

    computePrimalInitialPoint(initialScalings.omegaSquare, point);
    computeDualInitialPoint(initialScalings.omegaSquare, point);

    point.tau = 1.0;
    point.kappa = 1.0;

    return point;
  }

  /**
   * Computes primal point (x, s) from initial KKT matrix
   *
   * [d   A'  G'] [ x ]   [0]
   * [A  -d   0 ] [ y ] = [b]
   * [G   0  -I ] [-r ]   [h]
   *
   * s = r                 if r >k 0
   *   = r + (1 + alphaP)e otherwise
   */
  void computePrimalInitialPoint(const DenseVector& omegaSquare, Point& point) {
    DenseVector solution =
        findSolutionForRhs(omegaSquare, 0, _problem.b, _problem.h);

    // Copy subvectors
    point.x = blaze::subvector(solution, 0UL, _problem.columns);
    point.s =
        -blaze::subvector(solution, _problem.columns + _problem.equalityRows,
                          _problem.inequalityRows);

    bring2Cone(point.s);
  }

  /**
   * Computes dual point (y, z) from initial KKT matrix
   *
   * [d   A'  G'] [ x ]   [-c]
   * [A  -d   0 ] [ y ] = [ 0]
   * [G   0  -I ] [ z']   [ 0]
   *
   * z = z'                 if z' >k 0
   *   = z' + (1 + alphaD)e otherwise
   */
  void computeDualInitialPoint(const DenseVector& omegaSquare, Point& point) {
    DenseVector solution = findSolutionForRhs(omegaSquare, -_problem.c, 0, 0);

    // Copy subvectors
    point.y =
        blaze::subvector(solution, _problem.columns, _problem.equalityRows);
    point.z =
        blaze::subvector(solution, _problem.columns + _problem.equalityRows,
                         _problem.inequalityRows);

    bring2Cone(point.z);
  }

  /**
   * Move slack variables into Cone (Positive orthant)
   */
  void bring2Cone(DenseVector& vector) const {
    const double min = blaze::min(vector);
    // FIXME Use step length as this constant
    double alpha = -0.99;

    if (min < 0) {
      alpha = std::abs(min);
    }

    // Move point to positive cone
    for (size_t j = 0; j < vector.size(); ++j) {
      vector[j] += 1 + alpha;
    }
  }

  /**
   * FB, SB, TB means first block, second block, third block
   *
   * Creates RHS, finds solution and finally does iterative refinement to
   *improve solution
   */
  template <typename FB, typename SB, typename TB>
  DenseVector findSolutionForRhs(const DenseVector& omegaSquare, const FB& fb,
                                 const SB& sb, const TB& tb) {
    DenseVector rhs = createRHS(fb, sb, tb);

    DenseVector solution =
        doIterativeRefinement(omegaSquare, rhs, _lSolver.solveForRhs(rhs));
    return solution;
  };

  /**
   * FB, SB, TB means first block, second block, third block
   * TODO Use SplitVector, but rhs created here should not be constant think...
   */
  template <typename FB, typename SB, typename TB>
  DenseVector createRHS(const FB& fb, const SB& sb, const TB& tb) const {
    // Vector size is equal to size of KKT matrix
    DenseVector rhs(_problem.columns + _problem.equalityRows +
                    _problem.inequalityRows);

    blaze::DenseSubvector<DenseVector> firstBlock =
        blaze::subvector(rhs, 0UL, _problem.columns);
    firstBlock = fb;

    blaze::DenseSubvector<DenseVector> secondBlock =
        blaze::subvector(rhs, _problem.columns, _problem.equalityRows);
    secondBlock = sb;

    blaze::DenseSubvector<DenseVector> thirdBlock = blaze::subvector(
        rhs, _problem.columns + _problem.equalityRows, _problem.inequalityRows);
    thirdBlock = tb;

    return rhs;
  }

  /**
   * Iterative refinement
   * omegaSquare is only variable part, all other blocks in KKT matrix are
   *constant
   * omegaSquare is NSD
   *  [ d   A'  G'  ]
   *  [ A  -d   0   ]
   *  [ G   0  -W^2 ]
   *
   * TODO Commented prevSolution copy to improve speed, as this is a rare case,
   *better calculate rather than copy which I am doing here, In future calculate
   *prevSolution, for now throw exception
   */
  DenseVector doIterativeRefinement(const DenseVector& omegaSquare,
                                    const DenseVector& rhs,
                                    const DenseVector& solution) {
    if (!_problem.options.solverIR) {
      return solution;
    }
    // TODO Is this right way to represent nan?
    double prevError = std::nan("1");
    double errorThreshold = (1 + _kktUtil.nnz) * _problem.options.LSAcc;

    DenseVector newSolution = solution;
    //    DenseVector prevSolution;

    for (int j = 0; j < _problem.options.IRIterations; ++j) {
      const ResidualsKkt residuals(_problem, rhs, newSolution, omegaSquare);

      double errorNorm =
          normInf(residuals.kktX, residuals.kktY, residuals.kktZ);

      _logger->info("Error norm: {}, during iteration: {}", errorNorm, j);

      if (j > 0 && errorNorm > prevError) {
        _logger->info(
            "As Error norm is: {}, during iteration: {}, returning previous "
            "solution",
            errorNorm, j);
        //        return prevSolution;
        throw new std::invalid_argument("Iterative refinement failed");
      }

      if (errorNorm < errorThreshold ||
          (j > 0 && prevError < _problem.options.IRFactor * errorNorm)) {
        return newSolution;
      }

      prevError = errorNorm;
      //      prevSolution = newSolution;
      newSolution = newSolution +
                    _lSolver.solveForRhs(createRHS(
                        residuals.kktX, residuals.kktY, residuals.kktZ));
    }

    return newSolution;
  }

  /**
   * FIXME Code can be improved for both affine and combined methods
   */
  Point getAffineDirection(const Point& currentPoint,
                           const Residuals& residuals,
                           const NTScalings& scalings, const SplitVector& ds1,
                           const double& tauDenominator) {
    DenseVector wholeDs2 =
        findSolutionForRhs(scalings.omegaSquare, residuals.rx, residuals.ry,
                           -residuals.rz + currentPoint.s);

    SplitVector ds2(_problem, wholeDs2);

    Point affinePoint(_problem);

    affinePoint.tau =
        (residuals.rTau - currentPoint.kappa +
         blaze::trans(_problem.c) * ds2.x + blaze::trans(_problem.b) * ds2.y +
         blaze::trans(_problem.h) * ds2.z) /
        tauDenominator;

    affinePoint.x = ds2.x + affinePoint.tau * ds1.x;
    affinePoint.y = ds2.y + affinePoint.tau * ds1.y;
    affinePoint.z = ds2.z + affinePoint.tau * ds1.z;

    // deltaS = -W(lambda\ lambda o lambda + W*deletaZ )
    // -W^2 as we have added negative in scalings computation
    affinePoint.s = -currentPoint.s + scalings.omegaSquare * affinePoint.z;

    affinePoint.kappa =
        -currentPoint.kappa * (1 + affinePoint.tau / currentPoint.tau);

    return affinePoint;
  }

  /**
   * Only for LP cone
   */
  Point getCombinedDirection(const Point& currentPoint,
                             const Point& affinePoint,
                             const Residuals& residuals,
                             const NTScalings& scalings, const SplitVector& ds1,
                             const double& tauDenominator) {
    double sigma = computeSigma(computeAlpha(currentPoint, affinePoint));
    double oneMinusSigma = 1 - sigma;
    // (s' * z + kappa * tau) / D + 1
    double mu = (residuals.gap + (currentPoint.kappa * currentPoint.tau)) /
                (_problem.inequalityRows + 1);

    // Find RHS for combined direction
    DenseVector rx = oneMinusSigma * residuals.rx;
    DenseVector ry = oneMinusSigma * residuals.ry;
    // lambda o lambda + (W^-1 * delataSa) o (W * delataZa) - sigma * mu * e;
    DenseVector rs(affinePoint.s.size());
    for (size_t j = 0; j < rs.size(); ++j) {
      rs[j] = scalings.lambdaSquare[j] + (affinePoint.s[j] * affinePoint.z[j]) -
              (sigma * mu);
    }

    // -(1 - sigma)*rz + W(lambda \ ds)
    // Negative sign as W^2 has minus in scalings
    DenseVector rz(residuals.rz);
    for (size_t j = 0; j < rz.size(); ++j) {
      rz[j] = -oneMinusSigma * residuals.rz[j] + (rs[j] / currentPoint.z[j]);
    }

    // Get solution for RHS
    DenseVector wholeDs3 = findSolutionForRhs(scalings.omegaSquare, rx, ry, rz);

    SplitVector ds3(_problem, wholeDs3);

    Point combinedPoint(_problem);

    double rKappa = currentPoint.tau * currentPoint.kappa +
                    affinePoint.tau * affinePoint.kappa - sigma * mu;
    combinedPoint.tau =
        (oneMinusSigma * residuals.rTau - (rKappa / currentPoint.tau) +
         blaze::trans(_problem.c) * ds3.x + blaze::trans(_problem.b) * ds3.y +
         blaze::trans(_problem.h) * ds3.z) /
        tauDenominator;

    combinedPoint.x = ds3.x + combinedPoint.tau * ds1.x;
    combinedPoint.y = ds3.y + combinedPoint.tau * ds1.y;
    combinedPoint.z = ds3.z + combinedPoint.tau * ds1.z;

    // -W(lambda \ bs + W*delataZc)
    // Only for LP cone
    // Aware of sign in front of W^2
    for (size_t k = 0; k < rs.size(); ++k) {
      combinedPoint.s[k] = -rs[k] / currentPoint.z[k] +
                           scalings.omegaSquare[k] * combinedPoint.z[k];
    }

    combinedPoint.kappa =
        -(rKappa + currentPoint.kappa * combinedPoint.tau) / currentPoint.tau;

    return combinedPoint;
  }

  /**
   * line search only for positive orthant
   */
  double computeAlpha(const Point& currentPoint, const Point& searchDirection) {
    double rhoMin = searchDirection.s[0] / currentPoint.s[0];
    double sigmaMin = searchDirection.z[0] / currentPoint.z[0];
    double alpha;

    for (size_t j = 1; j < searchDirection.s.size(); ++j) {
      double rho = searchDirection.s[j] / currentPoint.s[j];
      double sigma = searchDirection.z[j] / currentPoint.z[j];

      if (rho < rhoMin) rhoMin = rho;

      if (sigma < sigmaMin) sigmaMin = sigma;
    }

    if (sigmaMin < rhoMin) {
      alpha = sigmaMin < 0 ? 1.0 / (-sigmaMin) : 1.0 / _problem.options.epsilon;
    } else {
      alpha = rhoMin < 0 ? 1.0 / (-rhoMin) : 1.0 / _problem.options.epsilon;
    }

    double tauBySearchTau = -currentPoint.tau / searchDirection.tau;
    double kappaBySearchKappa = -currentPoint.kappa / searchDirection.kappa;

    if (tauBySearchTau > 0 && tauBySearchTau < alpha) {
      alpha = tauBySearchTau;
    }
    if (kappaBySearchKappa > 0 && kappaBySearchKappa < alpha) {
      alpha = kappaBySearchKappa;
    }

    if (alpha > _problem.options.stepMax) alpha = _problem.options.stepMax;

    if (alpha < _problem.options.stepMin) alpha = _problem.options.stepMin;

    return alpha;
  }

  /**
   *
   */
  double computeSigma(double alpha) {
    double sigma = std::pow(1 - alpha, 3);

    if (sigma > _problem.options.sigmaMax) sigma = _problem.options.sigmaMax;
    if (sigma < _problem.options.sigmaMin) sigma = _problem.options.sigmaMin;

    return sigma;
  }
};

}  // lp
#endif  // SOLVER_HPP
