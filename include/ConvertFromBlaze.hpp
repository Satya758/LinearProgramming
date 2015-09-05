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
 */
template <typename ColumnPointer, typename RowIndex, typename RowValue>
void createUTCCSMatrix(const SymmetricMatrix& bMatrix, ColumnPointer* const cp,
                       RowIndex* ri, RowValue* rv) {
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

      // If row index goes beyond diagonal element stop the loop, as we are only
      // looking at upper triangular matrix
      if (colIter->index() >= i) {
        break;
      }
    }
    cp[i + 1] = columnPtr;
  }
}

/**
 * Returns nonzeros in upper triangle of given symmetric matrix
 */
size_t getSymmetricUtNonZeros(const SymmetricMatrix& bMatrix) {
  // As its guaranteed that there are diagonal elements, subtract rows/columns
  // (which is equal to diagonal elements) from overall nnz
  size_t strictlyUpperNnz = (bMatrix.nonZeros() - bMatrix.rows()) / 2;
  // Add back diagonal elements to get upper triangle nnz
  return strictlyUpperNnz + bMatrix.rows();
}

}  // lp
#endif  // CONVERTFROMBLAZE_HPP
