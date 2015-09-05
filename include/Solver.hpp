//
// Created by satya on 5/9/15.
//

#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "spdlog/spdlog.h"

#include "Problem.hpp"
#include "Point.hpp"

namespace lp {

template <typename LinearSolver>
class Solver {
 public:
  Solver(const Problem& problem)
      : _problem(problem), _logger(spdlog::stdout_logger_mt("Solver")) {}

  void solve() {
    _logger->info("Solver started");

    SymmetricMatrix kkt = createInitialKKT();

    SymmetricMatrix::ConstIterator iter = blaze::cbegin(kkt, 0);
    //    std::cout << kkt << std::endl;

    LinearSolver lSolver(kkt);
  }

 private:
  const Problem& _problem;
  const std::shared_ptr<spdlog::logger> _logger;

  //  Point getInitialPoint() {}

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
   * FIXME If there are no computations using Symmetric matrix Chnage to Upper
   *triangular matrix to save space
   */
  SymmetricMatrix createInitialKKT() {
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
};

}  // lp
#endif  // SOLVER_HPP
