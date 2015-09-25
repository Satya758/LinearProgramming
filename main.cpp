#include <iostream>

#include <parser/Parser.hpp>

#include "Solver.hpp"
#include "CholeskyLDLTSolver.hpp"

using namespace std;
/**
 *
 */
int main(int argc, char **argv) {
  //  lpp::Parser parser("/home/satya/LPProblems/beer.lp", true);
    lpp::Parser parser("/home/satya/LPProblems/test/dfl001.lp", true);
//  lpp::Parser parser("/home/satya/LPProblems/test/25fv47.lp", true);

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
