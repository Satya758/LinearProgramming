//
// Created by satya on 5/9/15.
//

#ifndef CHOLESKYLDLTSOLVER_HPP
#define CHOLESKYLDLTSOLVER_HPP

#include <cholmod.h>

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
   * SymmetricMatrix: Only matrix that is to be solved is provided. Matrix
   *provided in constructor is KKT matrix used to compute initial point to start
   *solver.
   * Linear solver can use this initial matrix to do symbolic analysis, as
   *non-zero pattern would not vary for this algorithm.
   * Uses CHOLMOD, incremental factorization
   */
  CholeskyLDLTSolver(const Problem& problem, const KKTUtil& kktUtil)
      : _problem(problem), _kktUtil(kktUtil) {
    cholmod_l_start(&c);

    // TODO For now its just AMD, have to try out metis and others as we have
    // Sparse G (Special) Lets use latest metis and define custom permutation in
    // cholmod
    c.nmethods = 1;
    // As we are using incremental factorization, permutation of original matrix
    // alters actual matrix that is factorized, so we have to ignore ordering
    // methods, TODO Unless there is a way to partial order without disturbing
    // 3X3 block, its either reduce filling ordering or incremental
    // factorization
    c.method[0].ordering = CHOLMOD_NATURAL;
    c.postorder = false;
    // c.method[0].ordering = CHOLMOD_AMD;

    c.itype = CHOLMOD_LONG;
    c.dtype = CHOLMOD_DOUBLE;

    // Keep diagonal elements in given -bound <= D <= bound
    c.dbound = _problem.options.dynamicDelta;

    // TODO Remove after testing
    // Print warning messages
    c.print = 3;
  }

  ~CholeskyLDLTSolver() {
    cholmod_l_free_sparse(&_A, &c);
    cholmod_l_free_factor(&_L, &c);
    cholmod_l_free_factor(&_PL, &c);

    cholmod_l_finish(&c);
  }

  /**
   * As this is first time this is complete factorization with symbolic analysis
   */
  void factorizeInitialMatrix(const DenseVector& omegaSquare) {
    // A is symmetric matrix, 6th parameter indicates that A is symmetric
    _A = cholmod_l_allocate_sparse(_kktUtil.size, _kktUtil.size, _kktUtil.utNnz,
                                   true, true, 1, CHOLMOD_REAL, &c);

    createInitialKktUtCcsMatrix(
        omegaSquare, static_cast<SuiteSparse_long*>(_A->p),
        static_cast<SuiteSparse_long*>(_A->i), static_cast<double*>(_A->x));

    // TODO Check if created matrix is good, remove it later after testing
    cholmod_l_print_sparse(_A, "A", &c);

    // Symbolic analysis
    _PL = cholmod_l_analyze(_A, &c);
    _PL->is_ll = false;
    _PL->is_super = false;

    factorize(0, _problem.columns + _problem.equalityRows, _PL);

    // Clean old factor before creating new one
    //    cholmod_l_free_factor(&_L, &c);
    // FIXME Instead of copying we can clean _PL to be used for subsequent
    // factorizations
    // TODO Meanwhile move to method
    _L = cholmod_l_copy_factor(_PL, &c);
    factorize(_problem.columns + _problem.equalityRows, _A->nrow, _L);

    cholmod_l_print_factor(_PL, "PL: ", &c);
    cholmod_l_print_factor(_L, "L: ", &c);
  }

  /**
   * Incremental factorization
   */
  void factorizeMatrix(const DenseVector& omegaSquare) {
    updateKktUtCcs3X3Block(omegaSquare, static_cast<SuiteSparse_long*>(_A->p),
                           static_cast<SuiteSparse_long*>(_A->i),
                           static_cast<double*>(_A->x));

    // Clean old factor before creating new one
    cholmod_l_free_factor(&_L, &c);
    _L = cholmod_l_copy_factor(_PL, &c);
    // After first factor only 3X3 diagonal block is changed so we can do
    // incremental factorization
    factorize(_problem.columns + _problem.equalityRows, _A->nrow, _L);
  }

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

  cholmod_factor* _L;
  cholmod_factor* _PL;  // Partial factor

  cholmod_common c;

  /**
   * FIXME use beta[0] instead of adding delta explicitly
   */
  void factorize(size_t startingIndex, size_t rows, cholmod_factor* _FL) {
    double beta[2];
    beta[0] = 0;
    beta[1] = 0;

    // endIndex is not actual index but number of rows, rows - 1 is done inside
    // to get index, read the doc!!!
    cholmod_l_rowfac(_A, nullptr, beta, startingIndex, rows, _FL, &c);
  }

  /**
   * Initial KKT Matrix, only upper triangle is filled to create Colum
   *compressed storage (CCS) sparse matrix
   *
   *  [ d   A'  G']
   *  [ A  -d   0 ]
   *  [ G   0  -I ]
   *
   * CCS arrays are allocated
   */
  template <typename ColumnPointer, typename RowIndex, typename RowValue>
  void createInitialKktUtCcsMatrix(const DenseVector& omegaSquare,
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
      ri[rowIndex] = k + _problem.columns + _problem.equalityRows;
      rv[rowIndex++] = omegaSquare[k];

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
   * Above condition is not true for Second order cones
   *
   * Only rowValue array is changed
   * Only 3X3 diagonal block is changed
   *
   */
  template <typename ColumnPointer, typename RowIndex, typename RowValue>
  void updateKktUtCcs3X3Block(const DenseVector& omegaSquare,
                              const ColumnPointer* const cp,
                              const RowIndex* const ri,
                              RowValue* const rv) const {
    size_t colIndex = _problem.columns + _problem.equalityRows;
    size_t columns =
        _problem.columns + _problem.equalityRows + _problem.inequalityRows;

    size_t scalingIndex = 0;
    // 3X3 block diagonal, scalings matrix
    for (size_t j = colIndex; j < columns; ++j) {
      // TODO Ordering is not done in favour of incremental factorization, so
      // IPerm is not used
      // j is actual column index and colI is permuted col index
      // size_t colI = IPerm[j];
      rv[cp[j + 1] - 1] = omegaSquare[scalingIndex++];
    }
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
