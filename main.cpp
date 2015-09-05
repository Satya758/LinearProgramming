#include <iostream>

#include <parser/Parser.hpp>

#include "Solver.hpp"
#include "CholeskyLDLTSolver.hpp"

using namespace std;
/**
 *
 */
int main(int argc, char **argv) {
  std::cout << "Hello, world!" << std::endl;

  lpp::Parser parser("/home/satya/LPProblems/QiTest.lp", true);

  lpp::Problem parserProblem = parser.getBlazeProblem();

  lp::Problem problem(parserProblem.equalityRows, parserProblem.inequalityRows,
                      parserProblem.columns);
  problem.c = parserProblem.c;

  problem.A = parserProblem.A;
  problem.b = parserProblem.b;

  problem.G = parserProblem.G;
  problem.h = parserProblem.h;

  lp::Solver<lp::CholeskyLDLTSolver> solver(problem);
  solver.solve();

  return 0;
}
