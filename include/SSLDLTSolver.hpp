//
// Created by satya on 5/9/15.
//

#ifndef CHOLESKYLDLTSOLVER_HPP
#define CHOLESKYLDLTSOLVER_HPP

#include <cholmod.h>
#include <amd.h>

#include "Problem.hpp"
#include "NTScalings.hpp"
#include "KKTUtil.hpp"

namespace lp {

/**
 * FIXME Error messages from cholmod are ignored!!!
 */
class CholeskyLDLTSolver {
 public:
  /**
   * Setup environment
   */
  CholeskyLDLTSolver(const Problem& problem, const KKTUtil& kktUtil)
      : _problem(problem),
        _kktUtil(kktUtil),
        _logger(spdlog::stdout_logger_mt("Cholmod")) {
    cholmod_l_start(&c);

    c.nmethods = 1;
    c.postorder = true;
    c.method[0].ordering = CHOLMOD_AMD;

    c.itype = CHOLMOD_LONG;
    c.dtype = CHOLMOD_DOUBLE;

    // Keep diagonal elements in given -bound <= D <= bound
    c.dbound = _problem.options.dynamicDelta;

    // TODO Remove after testing
    c.print = 3;
  }

  /**
   * Clean
   */
  ~CholeskyLDLTSolver() {
    cholmod_l_free_sparse(&_A, &c);

    cholmod_l_free_factor(&_L, &c);

    cholmod_l_finish(&c);
  }

  /**
   * SymmetricMatrix: Only matrix that is to be solved is provided. Matrix
   *provided in constructor is KKT matrix used to compute initial point to start
   *solver.
   * Linear solver can use this initial matrix to do symbolic analysis, as
   *non-zero pattern would not vary for this algorithm.
   *
   * [d   A'  G']
   * [A  -d   0 ]
   * [G   0  -I ]
   */
  void factorizeInitialMatrix(const NTScalings& scalings) {
    // A is symmetric matrix, 6th parameter indicates that A is symmetric
    _A = cholmod_l_allocate_sparse(_kktUtil.size, _kktUtil.size, _kktUtil.utNnz,
                                   true, true, 1, CHOLMOD_REAL, &c);

    createInitialKktUtCcsMatrix(scalings, static_cast<SuiteSparse_long*>(_A->p),
                                static_cast<SuiteSparse_long*>(_A->i),
                                static_cast<double*>(_A->x));

    // TODO Check if created matrix is good, remove it later after testing
    cholmod_l_print_sparse(_A, "A", &c);

    // Symbolic analysis
    _L = cholmod_l_analyze(_A, &c);
    _L->is_ll = false;
    _L->is_super = false;

    factorize(true);
  }

  /**
   * Incremental factorization
   */
  void factorizeMatrix(const NTScalings& scalings) {
    updateKktUtCcs3X3Block(scalings, static_cast<SuiteSparse_long*>(_A->p),
                           static_cast<SuiteSparse_long*>(_A->i),
                           static_cast<double*>(_A->x));

    factorize(false);
  }

  /**
   * Solve for Rhs
   */
  DenseVector solveForRhs(const DenseVector& rhs) {
    auto cholmod_del = [&](cholmod_dense* d) {
      // As owner of value pointers is transferred to Blaze, its responsibility
      // of blaze to delete this pointers
      d->x = nullptr;
      cholmod_l_free_dense(&d, &this->c);
    };

    std::unique_ptr<cholmod_dense, decltype(cholmod_del)> r(
        cholmod_l_allocate_dense(rhs.size(), 1, rhs.size(), CHOLMOD_REAL, &c),
        cholmod_del);

    // Copy of pointer not data be careful
    // Removing constantness as x is not
    r->x = const_cast<double*>(getDenseVector(rhs));

    std::unique_ptr<cholmod_dense, decltype(cholmod_del)> s(
        cholmod_l_solve(CHOLMOD_A, _L, r.get(), &c), cholmod_del);

    // Copy of pointer not data be careful not to delete it here
    DenseVector solution(s->nrow, static_cast<double*>(s->x));

    return solution;
  }

 private:
  const Problem& _problem;
  const KKTUtil& _kktUtil;

  // Copy contents of blaze matrix into cholmod sparse as blaze structures are
  // not incompatible with cholmod
  cholmod_sparse* _A;

  cholmod_factor* _L = nullptr;

  cholmod_common c;

  const std::shared_ptr<spdlog::logger> _logger;

  std::function<void(cholmod_sparse*)> cholmodSparseDelete =
      [this](cholmod_sparse* S) { cholmod_l_free_sparse(&S, &(this->c)); };

  using CholmodSparse =
      std::unique_ptr<cholmod_sparse, decltype(cholmodSparseDelete)>;

  /**
   * Initial KKT Matrix, only upper triangle is filled to create Colum
   *compressed storage (CCS) sparse matrix
   *
   *  [ d   A'  G']
   *  [ A  -d   0 ]
   *  [ G   0  -d ]
   *
   * CCS arrays are allocated
   */
  template <typename ColumnPointer, typename RowIndex, typename RowValue>
  void createInitialKktUtCcsMatrix(const NTScalings& scalings,
                                   ColumnPointer* const cp, RowIndex* const ri,
                                   RowValue* const rv) {
    // Nature of column pointer, always starts with 0 and ends with nnz
    cp[0] = 0;
    size_t columnPtr = 0;

    const SparseMatrix& AT = blaze::trans(_problem.A);
    const SparseMatrix& GT = blaze::trans(_problem.G);

    // Fill 1X1 diagonal block
    for (size_t j = 0; j < _problem.columns; ++j) {
      // As its first diagonal both index and rowIndex are same
      ri[j] = j;
      rv[j] = _problem.options.staticDelta;

      cp[j + 1] = ++columnPtr;
    }

    // As 1X1 diagonal which has columns entries is already filled, we start
    // next index from here
    size_t rowIndex = _problem.columns;
    // Fill 1X2, half of 2X2 block
    // Blaze Sparse matrix is CCS format so accessing columns is faster
    for (size_t k = 0; k < AT.columns(); ++k) {
      for (SparseMatrix::ConstIterator colIter = AT.cbegin(k);
           colIter != AT.cend(k); ++colIter) {
        ri[rowIndex] = colIter->index();
        rv[rowIndex++] = colIter->value();

        ++columnPtr;
      }

      // Add diagonal -delta
      ri[rowIndex] = k + _problem.columns;
      rv[rowIndex++] = -_problem.options.staticDelta;

      ++columnPtr;

      cp[_problem.columns + k + 1] = columnPtr;
    }

    // Fill 1X3 and half of 3X3 block
    for (size_t k = 0; k < GT.columns(); ++k) {
      for (SparseMatrix::ConstIterator colIter = GT.cbegin(k);
           colIter != GT.cend(k); ++colIter) {
        ri[rowIndex] = colIter->index();
        rv[rowIndex++] = colIter->value();

        ++columnPtr;
      }

      // Add diagonal omegaSquare at 3X3 block
      // Its -W^2, scalings has already added negative sign
      ri[rowIndex] = k + _problem.columns + _problem.equalityRows;
      rv[rowIndex++] = scalings.omegaSquare[k];

      ++columnPtr;

      cp[_problem.columns + _problem.equalityRows + k + 1] = columnPtr;
    }
  }

  /**
   * As we have only upper triangle stored in CCS format, we are sure that last
   * element in each column is diagonal element.
   * So Diagonal element can be accessed by rv[cp[i+1] - 1] of column i and row
   *i,
   * this is because of being diagonal element last
   * Above condition is not true for Second order cones (as its not just diagonal
   *change)
   *
   * Only rowValue array is changed
   * Only 3X3 diagonal block is changed
   *
   */
  template <typename ColumnPointer, typename RowIndex, typename RowValue>
  void updateKktUtCcs3X3Block(const NTScalings& scalings,
                              const ColumnPointer* const cp,
                              const RowIndex* const ri,
                              RowValue* const rv) const {
    size_t colIndex = _problem.columns + _problem.equalityRows;

    size_t scalingIndex = 0;
    // 3X3 block diagonal, scalings matrix
    for (size_t j = colIndex; j < _kktUtil.size; ++j) {
      rv[cp[j + 1] - 1] = scalings.omegaSquare[scalingIndex++];
    }
  }

  /**
   * FIXME use beta[0] instead of adding delta explicitly
   */
  void factorize(bool permute) {
    _logger->info("Factorization started");
    if (permute) {
      permuteMatrix();
    }

    double beta[2];
    beta[0] = 0;
    beta[1] = 0;

    cholmod_l_rowfac(_A, nullptr, beta, 0, _A->nrow, _L, &c);
    _logger->info("Factorization ended");
  }

  /**
   * Called only once during initial factorization, later factorizations use
   * permuted matrix to factorize
   */
  void permuteMatrix() {
    // Temporary holder
    // though stype is of no use for B matrix, given as input in this case to
    // make intention more clear
    cholmod_sparse* B = cholmod_l_allocate_sparse(
        _A->nrow, _A->ncol, _A->nzmax, true, true, -1, CHOLMOD_REAL, &c);
    // 1 in second parameter means normal array transpose (look into cholmod
    // doc)
    cholmod_l_transpose_sym(_A, 1, static_cast<SuiteSparse_long*>(_L->Perm), B,
                            &c);
    // transpose again to get upper triangular matrix
    cholmod_l_transpose_sym(B, 1, nullptr, _A, &c);

    // Free B matrix, we can use unique_ptr with custom deleter, but method is
    // so small its ok
    cholmod_l_free_sparse(&B, &c);
  }

  /**
   * Lifetime of pointer is maintained by blaze, keep that in mind, as we are in
   * scope of solver, pointer do not go out of scope
   */
  const double* getDenseVector(const DenseVector& bVec) const {
    return bVec.data();
  }
};

}  // lp
#endif  // CHOLESKYLDLTSOLVER_HPP
