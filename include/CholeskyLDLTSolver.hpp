//
// Created by satya on 5/9/15.
//

#ifndef CHOLESKYLDLTSOLVER_HPP
#define CHOLESKYLDLTSOLVER_HPP

#include <cholmod.h>

#include "Problem.hpp"
#include "ConvertFromBlaze.hpp"

namespace lp {

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
  CholeskyLDLTSolver(const SymmetricMatrix& kkt) : _kkt(kkt) {
    // TODO For now its just AMD, have to try out metis and others as we have
    // Sparse G
    c.nmethods = 1;
    c.method[0].ordering = CHOLMOD_AMD;

    c.itype = CHOLMOD_LONG;

    _A = cholmod_l_allocate_sparse(kkt.rows(), kkt.columns(),
                                   getSymmetricUtNonZeros(_kkt), true, true, 1,
                                   CHOLMOD_REAL, &c);

    createUTCCSMatrix(_kkt, static_cast<SuiteSparse_long*>(_A->p),
                      static_cast<SuiteSparse_long*>(_A->i),
                      static_cast<double*>(_A->x));

    cholmod_l_print_sparse(_A, "Satya", &c);
  }

  ~CholeskyLDLTSolver() { cholmod_free_sparse(&_A, &c); }

 private:
  const SymmetricMatrix& _kkt;

  // Copy contents of blaze matrix into cholmod sparse as blaze strcutures are
  // not incompatible with cholmod
  cholmod_sparse* _A;

  cholmod_common c;
};

}  // lp
#endif  // CHOLESKYLDLTSOLVER_HPP
