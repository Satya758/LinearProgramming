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
   * SymmetricMatrix: Only matrix that is to be solved is provided. Matrix
   *provided in constructor is KKT matrix used to compute initial point to start
   *solver.
   * Linear solver can use this initial matrix to do symbolic analysis, as
   *non-zero pattern would not vary for this algorithm.
   * Uses CHOLMOD, incremental factorization
   */
  CholeskyLDLTSolver(const Problem& problem, const KKTUtil& kktUtil)
      : _problem(problem),
        _kktUtil(kktUtil),
        _logger(spdlog::stdout_logger_mt("Cholmod")) {
    cholmod_l_start(&c);

    // TODO For now its just AMD, have to try out metis and others as we have
    // Sparse G (Special) Lets use latest metis and define custom permutation in
    // cholmod
    // FIXME Add support for more permutation options
    c.nmethods = 1;
    c.postorder = true;
    c.method[0].ordering = CHOLMOD_AMD;

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

    cholmod_l_free_factor(&_IL, &c);
    cholmod_l_free_factor(&_L, &c);

    cholmod_l_finish(&c);
  }

  /**
   * As this is first time this is complete factorization with symbolic analysis
   * First factorize following matrix
   * [d   A'  G']
   * [A  -d   0 ]
   * [G   0  -d ]
   *
   * and then using rank-k updates procedure update factor in this case
   * [d   A'  G']
   * [A  -d   0 ]
   * [G   0  -I ]
   */
  void factorizeInitialMatrix(const NTScalings& scalings) {
    // A is symmetric matrix, 6th parameter indicates that A is symmetric
    _A = cholmod_l_allocate_sparse(_kktUtil.size, _kktUtil.size, _kktUtil.utNnz,
                                   true, true, 1, CHOLMOD_REAL, &c);

    createInitialKktUtCcsMatrix(static_cast<SuiteSparse_long*>(_A->p),
                                static_cast<SuiteSparse_long*>(_A->i),
                                static_cast<double*>(_A->x));

    computePermutation(_A);

    // TODO Check if created matrix is good, remove it later after testing
    //    cholmod_l_print_sparse(_A, "A", &c);


    // Symbolic analysis
    _IL = cholmod_l_analyze(_A, &c);
    _IL->is_ll = false;
    _IL->is_super = false;

    _logger->info("First complete factorization started");
    cholmod_l_factorize(_A, _IL, &c);
    _logger->info("First complete factorization Ended");

    // TODO Remove
    //    cholmod_l_print_factor(_IL, "IL: ", &c);

    updateFactor(scalings.omega);
    _logger->info("Compute U Started");
    computeU(scalings.omega);
    _logger->info("Compute U Ended");
  }

  /**
   * Incremental factorization
   */
  void factorizeMatrix(const NTScalings& scalings) {
    updateFactor(scalings.omega);
    // TODO
    //    cholmod_l_print_factor(_L, "Second wave: ", &c);
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

  // Initial factor
  cholmod_factor* _IL;
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
  void createInitialKktUtCcsMatrix(ColumnPointer* const cp, RowIndex* const ri,
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
      rv[rowIndex++] = -_problem.options.staticDelta;

      ++columnPtr;

      cp[_problem.columns + _problem.equalityRows + k + 1] = columnPtr;
    }
  }

  /**
   *
   *
   */
  CholmodSparse createSparseUpdate(const DenseVector& omegaSquare) {
    // This is rectangular matrix of size _kktUtil.size X
    // _problem.inequalityRows, and as we are dealing with LP problem, number of
    // non-zeros are equal to _problem.inequalityRows, this is thin matrix
    CholmodSparse sparseUpdate(
        cholmod_l_allocate_sparse(_kktUtil.size, _problem.inequalityRows,
                                  _problem.inequalityRows, true, true, 0,
                                  CHOLMOD_REAL, &c),
        cholmodSparseDelete);

    SuiteSparse_long* Sp = static_cast<SuiteSparse_long*>(sparseUpdate->p);
    SuiteSparse_long* Si = static_cast<SuiteSparse_long*>(sparseUpdate->i);
    double* Sx = static_cast<double*>(sparseUpdate->x);

    size_t rowIndex = _problem.columns + _problem.equalityRows;
    Sp[0] = 0;
    for (size_t j = 0; j < sparseUpdate->ncol; ++j) {
      Sp[j + 1] = j + 1;
      Si[j] = rowIndex++;
      Sx[j] = omegaSquare[j];
    }
    // TODO
    //    cholmod_l_print_sparse(sparseUpdate.get(), "SU", &c);
    // Permute SparseUpdate
    CholmodSparse permutedSparseUpdate(
        cholmod_l_submatrix(sparseUpdate.get(),
                            static_cast<SuiteSparse_long*>(_L->Perm), _L->n,
                            nullptr, -1, true, true, &c),
        cholmodSparseDelete);

    return permutedSparseUpdate;
  }

  /**
   * Updates factor with rank-k update
   */
  void updateFactor(const DenseVector& omega) {
    _logger->info("Creation of duplicate copy started");
    // free _L before creating new one
    if (_L != nullptr) {
      cholmod_l_free_factor(&_L, &c);
    }
    _L = cholmod_l_copy_factor(_IL, &c);
    _logger->info("Creation of duplicate copy ended");

    _logger->info("Sparse update creation started");
    CholmodSparse sparseUpdate = createSparseUpdate(omega);
    _logger->info("Sparse update creation ended");

    _logger->info("factor update creation started");
    // Its downdate as omegaSquare is negative
    cholmod_l_updown(false, sparseUpdate.get(), _L, &c);
    _logger->info("factor update creation ended");
  }

  /**
   * In QU = R, compute U by repeated solves for each column in R, in this case
   * R = -W
   */
  void computeU(const DenseVector& omega) {
    cholmod_dense* x = nullptr;
    cholmod_dense* y = nullptr;
    cholmod_dense* e = nullptr;

    // Allocated only once, cleared every time in a loop for next use
    cholmod_dense* rhs = cholmod_l_zeros(_kktUtil.size, 1, CHOLMOD_REAL, &c);

    size_t rowIndex = _problem.columns + _problem.equalityRows;

    // Output pattern
    CholmodSparse Xset(nullptr, cholmodSparseDelete);

    // Instead of allocating every time, rewrite the values as there is only one
    // value
    CholmodSparse R(cholmod_l_allocate_sparse(_kktUtil.size, 1, 1, true, true,
                                              0, CHOLMOD_REAL, &c),
                    cholmodSparseDelete);
    // Just pattern is needed for R, values are ignored
    static_cast<SuiteSparse_long*>(R->p)[0] = 0;
    static_cast<SuiteSparse_long*>(R->p)[1] = 1;

    for (size_t j = 0; j < _problem.inequalityRows; ++j) {
      // Pattern information
      static_cast<SuiteSparse_long*>(R->i)[0] = rowIndex;
      // Actual value in dense vector
      static_cast<double*>(rhs->x)[rowIndex] = -omega[j];

      cholmod_sparse* XsetT = Xset.get();
      cholmod_l_solve2(CHOLMOD_A, _IL, rhs, R.get(), &x, &XsetT, &y, &e, &c);

      // Finally clear the added element in dense, to reuse same allocation in
      // next loop and increment rowIndex
      static_cast<double*>(rhs->x)[rowIndex++] = 0;
    }

    cholmod_l_free_dense(&rhs, &c);

    cholmod_l_free_dense(&x, &c);
    cholmod_l_free_dense(&y, &c);
    cholmod_l_free_dense(&e, &c);
  }

  /**
   * Computes partial permutations and combines them
   */
  void computePermutation(cholmod_sparse* A) {
    // First 2X2 block
    size_t faSize = _problem.columns + _problem.equalityRows;

//    CholmodSparse FA(
//        cholmod_l_allocate_sparse(faSize, faSize,
//                                  _problem.columns + _problem.A.nonZeros(),
//                                  true, true, 1, CHOLMOD_REAL, &c),
//        cholmodSparseDelete);
//
//    // Whole matrix without 2X2 block, we call it Second A
//    CholmodSparse SA(
//        cholmod_l_allocate_sparse(_kktUtil.size, _kktUtil.size,
//                                  _problem.inequalityRows + _problem.G.nonZeros(),
//                                  true, true, 1, CHOLMOD_REAL, &c),
//        cholmodSparseDelete);

    cholmod_sparse* FA = cholmod_l_allocate_sparse(faSize, faSize,
                                  _problem.columns + _problem.equalityRows + _problem.A.nonZeros(),
                                  true, true, 1, CHOLMOD_REAL, &c);

    cholmod_sparse* SA = cholmod_l_allocate_sparse(_kktUtil.size, _kktUtil.size,
                                  _problem.inequalityRows + _problem.G.nonZeros(),
                                  true, true, 1, CHOLMOD_REAL, &c);

    SuiteSparse_long* Ap = static_cast<SuiteSparse_long*>(A->p);
    SuiteSparse_long* Ai = static_cast<SuiteSparse_long*>(A->i);

    SuiteSparse_long* FAp = static_cast<SuiteSparse_long*>(FA->p);
    SuiteSparse_long* FAi = static_cast<SuiteSparse_long*>(FA->i);

    // TODO We can use std::copy
    for (size_t j = 0; j <= faSize; ++j) {
      FAp[j] = Ap[j];
    }

    for (size_t j = 0; j < FAp[faSize]; ++j) {
      FAi[j] = Ai[j];
    }

    SuiteSparse_long* SAp = static_cast<SuiteSparse_long*>(SA->p);
    SuiteSparse_long* SAi = static_cast<SuiteSparse_long*>(SA->i);

    for (size_t k = 0; k <= faSize ; ++k) {
      SAp[k] = 0;
    }

    for (size_t l = faSize + 1; l <= _kktUtil.size; ++l) {
     SAp[l] = Ap[l] - Ap[faSize];
    }

    size_t i = 0;
    for (size_t k = Ap[faSize]; k < Ap[_kktUtil.size]; ++k) {
      SAi[i] = Ai[k];
      i++;
    }

//    cholmod_l_print_sparse(A, "A", &c);
//    cholmod_l_print_sparse(FA.get(), "FA", &c);
//    cholmod_l_print_sparse(SA.get(), "SA", &c);

    _logger->info("Permuation strted 1");
    auto faPerm = std::make_unique<SuiteSparse_long[]>(FA->ncol);
    amd_l_order(FA->ncol, FAp, FAi, faPerm.get(), nullptr, nullptr);

    auto saPerm = std::make_unique<SuiteSparse_long[]>(SA->ncol);
    amd_l_order(SA->ncol, SAp, SAi, saPerm.get(), nullptr, nullptr);

    _logger->info("Permuation ended all");

    for (size_t m = 0; m < SA->ncol; ++m) {
      std::cout << saPerm[m] << std::endl;
    }
    cholmod_l_free_sparse(&FA, &c);
    cholmod_l_free_sparse(&SA, &c);
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
