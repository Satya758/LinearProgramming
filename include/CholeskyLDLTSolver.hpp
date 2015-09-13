//
// Created by satya on 5/9/15.
//

#ifndef CHOLESKYLDLTSOLVER_HPP
#define CHOLESKYLDLTSOLVER_HPP

#include <cholmod.h>

#include "Problem.hpp"
#include "NTScalings.hpp"
#include "ConvertFromBlaze.hpp"

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
  CholeskyLDLTSolver(const Problem& problem) : _problem(problem) {
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

    c.itype = CHOLMOD_LONG;
    c.dtype = CHOLMOD_DOUBLE;

    // Keep diagonal elements in given -bound <= D <= bound
    c.dbound = _problem.options.dynamicDelta;

    // TODO Remove after testing
    // Print warning messages
    c.print = 5;
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
  void factorizeInitialMatrix(const SymmetricMatrix& kkt) {
    // A is symmetric matrix, 6th parameter indicates that A is symmetric
    _A = cholmod_l_allocate_sparse(kkt.rows(), kkt.columns(),
                                   getSymmetricUtNnz(kkt), true, true, 1,
                                   CHOLMOD_REAL, &c);

    createUtCcsMatrix(kkt, static_cast<SuiteSparse_long*>(_A->p),
                      static_cast<SuiteSparse_long*>(_A->i),
                      static_cast<double*>(_A->x));

    // TODO Check if created matrix is good, remove it later after testing
    //    cholmod_l_check_sparse(_A, &c);

    // Symbolic analysis
    _PL = cholmod_l_analyze(_A, &c);
    _PL->is_ll = false;
    _PL->is_super = false;
    // computeIPerm();

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
  void factorizeMatrix(const NTScalings& scalings) {
    updateUtCcsLastBlock(scalings, _problem,
                         static_cast<SuiteSparse_long*>(_A->p),
                         static_cast<SuiteSparse_long*>(_A->i),
                         static_cast<double*>(_A->x), nullptr);

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
    // Natural ordering is used
    // permuteMatrix();

    double beta[2];
    beta[0] = 0;
    beta[1] = 0;

    // endIndex is not actual index but number of rows, rows - 1 is done inside
    // to get index, read the doc!!!
    cholmod_l_rowfac(_A, nullptr, beta, startingIndex, rows, _FL, &c);
  }

  /**
   * As whole _A is not copied again we have to permute only once!!
   * Natural ordering is used, so no use of this method
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
   * Compute Inverse permutation vector to update scalings
   *
   * Used to find permuted col/row from unpermuted col/row index
   *
   * Natural order is used, so no permutation required
   */
  void computeIPerm() {
    SuiteSparse_long* Perm = static_cast<SuiteSparse_long*>(_L->Perm);

    // allocate memory
    _L->IPerm = cholmod_l_malloc(_L->n, sizeof(SuiteSparse_long), &c);
    SuiteSparse_long* IPerm = static_cast<SuiteSparse_long*>(_L->IPerm);

    if (!IPerm) {
      // FIXME Error, raise exception, check Common.status for actual reason
      // Maybe out of memory
    }

    for (size_t j = 0; j < _L->n; ++j) {
      IPerm[Perm[j]] = j;
    }
  }
};

}  // lp
#endif  // CHOLESKYLDLTSOLVER_HPP
