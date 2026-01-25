#include <fstream>
#include <ilcplex/ilocplex.h>
#include <iomanip>
#include <iostream>
#include <vector>

ILOSTLBEGIN

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "2Usage: ./solve_tsfc instance.txt\n";
    return 1;
  }

  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "Cannot open file.\n";
    return 1;
  }

  int I, J, K;
  in >> I >> J >> K;

  std::vector<double> s(I), d(K);
  for (int i = 0; i < I; i++)
    in >> s[i];
  for (int k = 0; k < K; k++)
    in >> d[k];

  std::vector<std::vector<double>> b(I, std::vector<double>(J));
  std::vector<std::vector<double>> f(I, std::vector<double>(J));
  std::vector<std::vector<double>> c(J, std::vector<double>(K));
  std::vector<std::vector<double>> g(J, std::vector<double>(K));

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      in >> b[i][j];

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      in >> f[i][j];

  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++)
      in >> c[j][k];

  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++)
      in >> g[j][k];

  in.close();

  IloEnv env;
  try {
    IloModel model(env);

    /* VARIABLES */

    IloArray<IloNumVarArray> x(env, I);
    IloArray<IloNumVarArray> z(env, I);
    for (int i = 0; i < I; i++) {
      x[i] = IloNumVarArray(env, J, 0.0, IloInfinity, ILOFLOAT);
      z[i] = IloNumVarArray(env, J, 0.0, 1.0, ILOBOOL);
    }

    IloArray<IloNumVarArray> y(env, J);
    IloArray<IloNumVarArray> w(env, J);
    for (int j = 0; j < J; j++) {
      y[j] = IloNumVarArray(env, K, 0.0, IloInfinity, ILOFLOAT);
      w[j] = IloNumVarArray(env, K, 0.0, 1.0, ILOBOOL);
    }

    /* OBJECTIVE */

    IloExpr obj(env);

    for (int i = 0; i < I; i++)
      for (int j = 0; j < J; j++)
        obj += b[i][j] * x[i][j] + f[i][j] * z[i][j];

    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++)
        obj += c[j][k] * y[j][k] + g[j][k] * w[j][k];

    model.add(IloMinimize(env, obj));
    obj.end();

    /* CONSTRAINTS */

    // Plant capacity
    for (int i = 0; i < I; i++) {
      IloExpr expr(env);
      for (int j = 0; j < J; j++)
        expr += x[i][j];
      model.add(expr <= s[i]);
      expr.end();
    }

    // Customer demand
    for (int k = 0; k < K; k++) {
      IloExpr expr(env);
      for (int j = 0; j < J; j++)
        expr += y[j][k];
      model.add(expr == d[k]);
      expr.end();
    }

    // Flow conservation at depots
    for (int j = 0; j < J; j++) {
      IloExpr in(env), out(env);
      for (int i = 0; i < I; i++)
        in += x[i][j];
      for (int k = 0; k < K; k++)
        out += y[j][k];
      model.add(in == out);
      in.end();
      out.end();
    }

    // Linking x and z
    for (int i = 0; i < I; i++)
      for (int j = 0; j < J; j++)
        model.add(x[i][j] <= s[i] * z[i][j]);

    // Linking y and w
    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++)
        model.add(y[j][k] <= d[k] * w[j][k]);

    /* SOLVE */

    IloCplex cplex(model);

    // ---- LIMIT 2 SATA ----
    cplex.setParam(IloCplex::TiLim, 7200); // 2h
    cplex.setParam(IloCplex::Threads, 1);
    cplex.setOut(env.getNullStream());

    bool solved = cplex.solve();

    // Format ispisa za tabelu:
    // instanca  status  obj  time  iters  nodes

    std::cout << std::fixed << std::setprecision(2);

    if (solved) {

      IloAlgorithm::Status status = cplex.getStatus();

      double time = cplex.getTime();
      double iters = cplex.getNiterations();
      double nodes = cplex.getNnodes();

      if (cplex.isPrimalFeasible()) {
        double objVal = cplex.getObjValue();

        if (status == IloAlgorithm::Optimal) {
          // OPTIMALNO RESENJE
          std::cout << argv[1] << " "
                    << "OPTIMAL " << objVal << " " << time << " " << iters
                    << " " << nodes << "\n";
        } else {
          // DOPUSTIVO ALI NIJE OPTIMALNO
          std::cout << argv[1] << " "
                    << "NOT_OPTIMAL " << objVal << " " << time << " " << iters
                    << " " << nodes << "\n";
        }

      } else {
        // NEMA NI JEDNO DOPUSTIVO
        std::cout << argv[1] << " "
                  << "NO_FEASIBLE "
                  << "- " << time << " " << iters << " " << nodes << "\n";
      }

    } else {
      // Solver uopšte nije uspeo
      std::cout << argv[1] << " "
                << "NO_FEASIBLE "
                << "- " << cplex.getTime() << " " << cplex.getNiterations()
                << " " << cplex.getNnodes() << "\n";
    }

  } catch (IloException &e) {
    std::cerr << e << std::endl;
  }

  env.end();
  return 0;
}
