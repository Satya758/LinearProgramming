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

namespace lp {

template <typename LinearSolver>
class Solver {
 public:
  Solver(const Problem& problem)
      : _problem(problem),
        _lSolver(problem),
        _logger(spdlog::stdout_logger_mt("Solver")) {}

  void solve() {
    _logger->info("Solver started");

    SymmetricMatrix kkt = createInitialKKT();

    Point currentPoint = getInitialPoint(kkt);

    for (int j = 0; j < _problem.options.IRIterations; ++j) {
      // Compute residuals
      // Check for termination conditions
      const NTScalings scalings(_problem, currentPoint);

      updateKktWithNewScalings(scalings, kkt);
    }

    _logger->info("Solver ended");
  }

 private:
  const Problem& _problem;
  LinearSolver _lSolver;
  const std::shared_ptr<spdlog::logger> _logger;

  /**
   * Update KKT matrix with new NT scalings
   *
   * [d   A'  G']
   * [A   d   0 ]
   * [G   0  -W ]
   *
   * TODO Blaze symmetric matrix and chomod upper triangular matrices are
   *updated with same scaling matrix in two different places, how can we
   *streamline this?
   */
  void updateKktWithNewScalings(const NTScalings& scalings,
                                SymmetricMatrix& kkt) const {
    // 3X3 block diagonal, scalings matrix
    for (size_t j = _problem.columns + _problem.equalityRows; j < kkt.columns();
         ++j) {
      // TODO Minus before omega is easy to miss what to do
      kkt(j, j) = -scalings.omegaSquare[j];
    }
  }

  /**
   * Computes initial point from initial KKT matrix
   * [d   A'  G']
   * [A   d   0 ]
   * [G   0  -I ]
   */
  Point getInitialPoint(const SymmetricMatrix& initialKkt) {
    // Factorize Quasi PSD matrix
    _lSolver.factorizeInitialMatrix(initialKkt);
    // Create empty primal dual point
    Point point(_problem);

    computePrimalInitialPoint(initialKkt, point);
    computeDualInitialPoint(initialKkt, point);

    point.tau = 1.0;
    point.kappa = 1.0;

    return point;
  }

  /**
   * Computes primal point (x, s) from initial KKT matrix
   *
   * [d   A'  G'] [ x ]   [0]
   * [A   d   0 ] [ y ] = [b]
   * [G   0  -I ] [-r ]   [h]
   *
   * s = r                 if r >k 0
   *   = r + (1 + alphaP)e otherwise
   */
  void computePrimalInitialPoint(const SymmetricMatrix& initialKkt,
                                 Point& point) {
    DenseVector solution =
        findSolutionForRhs(initialKkt, 0, _problem.b, _problem.h);

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
   * [A   d   0 ] [ y ] = [ 0]
   * [G   0  -I ] [ z']   [ 0]
   *
   * z = z'                 if z' >k 0
   *   = z' + (1 + alphaD)e otherwise
   */
  void computeDualInitialPoint(const SymmetricMatrix& initialKkt,
                               Point& point) {
    DenseVector solution = findSolutionForRhs(initialKkt, -_problem.c, 0, 0);

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
  DenseVector findSolutionForRhs(const SymmetricMatrix& kkt, const FB& fb,
                                 const SB& sb, const TB& tb) {
    DenseVector rhs = createRHS(fb, sb, tb);

    DenseVector solution =
        doIterativeRefinement(kkt, rhs, _lSolver.solveForRhs(rhs));
    return solution;
  };

  /**
   *  Initial KKT Matrix
   *  [ d   A'  G']
   *  [ A   d   0 ]
   *  [ G   0  -I ]
   *
   *  Where d is static delta added along the diagonal to matrix to make quasi
   *definite
   *
   *  TODO Hopefully same matrix is used all along just by modifying last 3X3
   *block
   *
   * FIXME If there are no computations using Symmetric matrix change to Upper
   *triangular matrix to save space
   */
  SymmetricMatrix createInitialKKT() const {
    // Same value can be used for number of diagonal elements
    size_t kktMatrixSize =
        _problem.columns + _problem.equalityRows + _problem.inequalityRows;
    // Number of non zeros in KKT symmetric matrix, only lower or upper part is
    // used for count (which includes diagonal elements which are non-zero)
    // kktMatrixSize is used as number of diagonal elements
    // Blaze requires non-zeros of whole matrix rather than just lower/upper
    // matrix, so there is multiplication factor of 2
    size_t kktMatrixNnz =
        2 * (_problem.A.nonZeros() + _problem.G.nonZeros()) + kktMatrixSize;

    _logger->info("KKT matrix size: {}, KKT matrix nnz: {}", kktMatrixSize,
                  kktMatrixNnz);

    SymmetricMatrix kkt(kktMatrixSize);
    kkt.reserve(kktMatrixNnz);

    // Assign A
    blaze::submatrix(kkt, _problem.columns, 0UL, _problem.equalityRows,
                     _problem.columns) = _problem.A;
    // Assign G
    blaze::submatrix(kkt, _problem.columns + _problem.equalityRows, 0UL,
                     _problem.inequalityRows, _problem.columns) = _problem.G;

    // TODO Until blaze introduces identity matrices, use loop (as they are more
    // readable)
    // TODO Switch to blaze identity if they don't degrade performance
    // 1X1 block diagonal
    for (size_t j = 0; j < _problem.columns; ++j) {
      kkt(j, j) = _problem.options.staticDelta;
    }
    // 2X2 block, center piece diagonal
    for (size_t j = _problem.columns;
         j < _problem.columns + _problem.equalityRows; ++j) {
      kkt(j, j) = -_problem.options.staticDelta;
    }
    // 3X3 block diagonal, identity matrix
    for (size_t j = _problem.columns + _problem.equalityRows; j < kktMatrixSize;
         ++j) {
      kkt(j, j) = -1;
    }

    return kkt;
  }

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
   */
  DenseVector doIterativeRefinement(const SymmetricMatrix& kkt,
                                    const DenseVector& rhs,
                                    const DenseVector& solution) {
    double prevError = std::nan("1");
    double errorThreshold = (1 + kkt.nonZeros()) * _problem.options.LSAcc;

    DenseVector newSolution = solution;
    DenseVector prevSolution;

    for (int j = 0; j < _problem.options.IRIterations; ++j) {
      DenseVector residual = rhs - kkt * newSolution;
      double errorNorm = normInf(residual);

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
      newSolution = newSolution + _lSolver.solveForRhs(residual);
    }

    return newSolution;
  }
};

}  // lp
#endif  // SOLVER_HPP
