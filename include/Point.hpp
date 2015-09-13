//
// Created by satya on 5/9/15.
//

#ifndef POINT_HPP
#define POINT_HPP

#include "Problem.hpp"

namespace lp {
/**
 * Primal Dual point.
 */
class Point {
 public:
  Point(const Problem& problem)
      : x(problem.columns),
        s(problem.inequalityRows),
        y(problem.equalityRows),
        z(problem.inequalityRows) {}

  // Primal variables
  DenseVector x;
  DenseVector s;

  // Dual variables
  DenseVector y;
  DenseVector z;

  // Homogenizing variables
  double tau;
  double kappa;
};

/**
 *
 */
std::ostream& operator<<(std::ostream& out, const Point& point) {
  using namespace std;

  out << endl << "##################### Point Start" << endl;
  out << "Primal Variable x:" << endl << point.x << endl;
  out << "Primal Variable s:" << endl << point.s << endl;
  out << "Dual Variable y:" << endl << point.y << endl;
  out << "Dual Variable z:" << endl << point.z << endl;
  out << "Homogenizing Variable kappa: " << endl << point.kappa << endl;
  out << "Homogenizing Variable tau: " << endl << point.tau << endl;
  out << "##################### Point End" << endl;

  return out;
}

}  // lp
#endif  // POINT_HPP
