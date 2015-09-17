//
// Created by satya on 5/9/15.
//

#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "spdlog/spdlog.h"

#include "Problem.hpp"
#include "Point.hpp"
#include "LinearAlgebraUtil.hpp"
#include "NTScalings.hpp"
#include "KKTUtil.hpp"
#include "Residuals.hpp"

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

    // Initial omegaSquare
    // Initialize with negative one
    DenseVector initialOmegaSquare(_problem.inequalityRows, -1);

    Point currentPoint = getInitialPoint(initialOmegaSquare);

    for (int j = 0; j < 1; ++j) {
      // Compute residuals
      // Check for termination conditions
      const NTScalings scalings(_problem, currentPoint);

      _lSolver.factorizeMatrix(scalings.omegaSquare);

      DenseVector solution = findSolutionForRhs(
          scalings.omegaSquare, -_problem.c, _problem.b, _problem.h);

      std::cout << "First solution " << solution << std::endl;
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
  Point getInitialPoint(const DenseVector& omegaSquare) {
    // Factorize Quasi PSD matrix
    _lSolver.factorizeInitialMatrix(omegaSquare);
    // Create empty primal dual point
    Point point(_problem);

    computePrimalInitialPoint(omegaSquare, point);
    computeDualInitialPoint(omegaSquare, point);

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
   */
  DenseVector doIterativeRefinement(const DenseVector& omegaSquare,
                                    const DenseVector& rhs,
                                    const DenseVector& solution) {
    // TODO Is this right way to represent nan?
    double prevError = std::nan("1");
    double errorThreshold = (1 + _kktUtil.nnz) * _problem.options.LSAcc;

    DenseVector newSolution = solution;
    DenseVector prevSolution;

    for (int j = 0; j < _problem.options.IRIterations; ++j) {
      Residuals residuals(_problem, rhs, newSolution, omegaSquare);

      double errorNorm = normInf(residuals.x, residuals.y, residuals.z);

      _logger->info("Error norm: {}, during iteration: {}", errorNorm, j);

      if (j > 0 && errorNorm > prevError) {
        _logger->info(
            "As Error norm is: {}, during iteration: {}, returning previous "
            "solution",
            errorNorm, j);
        return prevSolution;
      }

      if (errorNorm < errorThreshold ||
          (j > 0 && prevError < _problem.options.IRFactor * errorNorm)) {
        return newSolution;
      }

      prevError = errorNorm;
      prevSolution = newSolution;
      newSolution = newSolution +
                    _lSolver.solveForRhs(
                        createRHS(residuals.x, residuals.y, residuals.z));
    }

    return newSolution;
  }
};

}  // lp
#endif  // SOLVER_HPP
