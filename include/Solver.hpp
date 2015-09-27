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

    for (int j = 0; j < 1; ++j) {
      const Residuals residuals(_problem, currentPoint, rxNorm, ryNorm, rzNorm);
      // Compute residuals
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

      //      std::cout << affinePoint << std::endl;
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
   *
   */
  Point getAffineDirection(const Point& currentPoint,
                           const Residuals& residuals,
                           const NTScalings& scalings, const SplitVector& ds1,
                           const double& tauDenominator) {
    DenseVector ds2 =
        findSolutionForRhs(scalings.omegaSquare, residuals.rx, residuals.ry,
                           -residuals.rz + currentPoint.s);

    SplitVector splitDs2(_problem, ds2);

    Point affinePoint(_problem);

    affinePoint.tau = (residuals.rTau - currentPoint.kappa +
                       blaze::trans(_problem.c) * splitDs2.x +
                       blaze::trans(_problem.b) * splitDs2.y +
                       blaze::trans(_problem.h) * splitDs2.z) /
                      tauDenominator;

    affinePoint.x = ds1.x + affinePoint.tau * splitDs2.x;
    affinePoint.y = ds1.y + affinePoint.tau * splitDs2.y;
    affinePoint.z = ds1.z + affinePoint.tau * splitDs2.z;

    // deltaS = -W(lambda\ lambda o lambda + W*deletaZ )
    // -W^2 as we have added negative in scalings computation
    affinePoint.s = currentPoint.s - scalings.omegaSquare * affinePoint.z;

    affinePoint.kappa =
        -currentPoint.kappa * (1 + affinePoint.tau / currentPoint.tau);

    return affinePoint;
  }
};

}  // lp
#endif  // SOLVER_HPP
