//
// Created by satya on 5/9/15.
//

#ifndef CONVERTFROMBLAZE_HPP
#define CONVERTFROMBLAZE_HPP

#include "Problem.hpp"
/**
 * These are helper functions used by linear solvers to convert blaze matrices
 * to CCS
 * (Compressed Column Storage) or CRS and vectors to plain arrays.
 *
 * FIXME Move to linear solver as free functions if another file feels is too
 *much
 */

namespace lp {

/**
 * Converts blaze sparse symmetric matrix to Upper Triangular CCS form.
 * Only upper triangular part of symmetric matrix is accessed.
 *
 * Though blaze symmetric matrix almost double the memory of upper triangular
 *matrix, its preferred as symmetric matrix is used in arithmetic operations
 *during iterative refinement
 *
 * Accepts arrays to insert values into, arrays are allocated with required
 *size requirements and type
 * ColumnPointer and RowIndex are array of ints and RowValue is array of double
 *
 * Creates copy of blaze matrix.
 *
 * Initial KKT Matrix
 *  [ d   A'  G']
 *  [ A  -d   0 ]
 *  [ G   0  -I ]
 */
template <typename ColumnPointer, typename RowIndex, typename RowValue>
void createUtCcsMatrix(const Problem& problem, const DenseVector& omegaSquare,
                       ColumnPointer* const cp, RowIndex* const ri,
                       RowValue* const rv) {
  // Nature of column pointer, always starts with 0 and ends with nnz
  cp[0] = 0;
  size_t columnPtr = 0;
  size_t rowIndex = 0;

  for (size_t i = 0; i < bMatrix.columns(); ++i) {
    for (SymmetricMatrix::ConstIterator colIter = bMatrix.cbegin(i);
         colIter != bMatrix.cend(i); ++colIter) {
      ++columnPtr;

      rv[rowIndex] = colIter->value();
      ri[rowIndex] = colIter->index();

      ++rowIndex;

      // If row index is equal or beyond diagonal element stop the loop, as we
      // are only
      // looking at upper triangular matrix.
      // As diagonal element is guaranteed to be present equality check is
      // enough though
      if (colIter->index() >= i) {
        break;
      }
    }
    cp[i + 1] = columnPtr;
  }
}

/**
 * As we have only upper triangle stored in CCS format, we are sure that last
 * element in each column is diagonal element.
 * So Diagonal element can be accessed by rv[cp[i+1] - 1] of column i and row i,
 * this is because of being diagonal element last
 * Above condition is not true for Second order cones
 *
 * Only rowValue array is changed
 * Only 3X3 diagonal block is changed (Which is already permuted)
 *
 * TODO This free function diviated from Blaze to CCS to Cholmod (Due to use of
 *permutation matrix), so question is does this method belong here or in
 *Choleksy solver?
 */
template <typename ColumnPointer, typename RowIndex, typename RowValue>
void updateUtCcsLastBlock(const NTScalings& scalings, const Problem& problem,
                          const ColumnPointer* const cp,
                          const RowIndex* const ri, RowValue* const rv,
                          const SuiteSparse_long* const IPerm) {
  size_t colIndex = problem.columns + problem.equalityRows;
  size_t columns =
      problem.columns + problem.equalityRows + problem.inequalityRows;

  size_t scalingIndex = 0;
  // 3X3 block diagonal, scalings matrix
  for (size_t j = colIndex; j < columns; ++j) {
    // TODO Ordering is not done in favour of incremental factorization, so
    // IPerm is not used
    // j is actual column index and colI is permuted col index
    // size_t colI = IPerm[j];
    // TODO Minus before omega is easy to miss what to do
    rv[cp[j + 1] - 1] = -scalings.omegaSquare[scalingIndex++];
  }
}

/**
 * Lifetime of pointer is maintained by blaze, keep that in mind, as we are in
 * scope of solver, pointer do not go out of scope
 */
const double* getDenseVector(const DenseVector& bVec) { return bVec.data(); }

}  // lp
#endif  // CONVERTFROMBLAZE_HPP
