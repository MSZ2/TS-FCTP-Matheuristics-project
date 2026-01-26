#include <fstream>
#include <ilcplex/ilocplex.h>
#include <iostream>
#include <iomanip>

ILOSTLBEGIN

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./solve_tsfc instance.txt\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "Cannot open file.\n";
        return 1;
    }

    // --- UCITAVANJE INSTANCE ---
    IloInt I, J, K; 
    in >> I >> J >> K;   // CHANGED: use IloInt instead of int

    IloNumArray s(I);    // CHANGED: use IloNumArray
    IloNumArray d(K);    // CHANGED: use IloNumArray

    for (IloInt i = 0; i < I; i++) in >> s[i];
    for (IloInt k = 0; k < K; k++) in >> d[k];

    // CHANGED: use IloArray<IloNumArray> instead of std::vector
    IloArray<IloNumArray> b(I), f(I);
    IloArray<IloNumArray> c(J), g(J);

    for (IloInt i = 0; i < I; i++) {
        b[i] = IloNumArray(J);
        f[i] = IloNumArray(J);
    }

    for (IloInt j = 0; j < J; j++) {
        c[j] = IloNumArray(K);
        g[j] = IloNumArray(K);
    }

    for (IloInt i = 0; i < I; i++)
        for (IloInt j = 0; j < J; j++) in >> b[i][j];

    for (IloInt i = 0; i < I; i++)
        for (IloInt j = 0; j < J; j++) in >> f[i][j];

    for (IloInt j = 0; j < J; j++)
        for (IloInt k = 0; k < K; k++) in >> c[j][k];

    for (IloInt j = 0; j < J; j++)
        for (IloInt k = 0; k < K; k++) in >> g[j][k];

    in.close();

    IloEnv env;
    try {
        IloModel model(env);

        // --- VARIJABLE ---
        IloNumVarArray x(env, I * J, 0.0, IloInfinity, ILOFLOAT);
        IloBoolVarArray z(env, I * J);   // CHANGED: use IloBoolVarArray
        IloNumVarArray y(env, J * K, 0.0, IloInfinity, ILOFLOAT);
        IloBoolVarArray w(env, J * K);   // CHANGED: use IloBoolVarArray

        auto idx_x = [J](IloInt i, IloInt j) { return i * J + j; };
        auto idx_y = [K](IloInt j, IloInt k) { return j * K + k; };

        // --- OBJEKTIVNA FUNKCIJA ---
        IloExpr obj(env);
        for (IloInt i = 0; i < I; i++)
            for (IloInt j = 0; j < J; j++)
                obj += b[i][j] * x[idx_x(i,j)] + f[i][j] * z[idx_x(i,j)];

        for (IloInt j = 0; j < J; j++)
            for (IloInt k = 0; k < K; k++)
                obj += c[j][k] * y[idx_y(j,k)] + g[j][k] * w[idx_y(j,k)];

        model.add(IloMinimize(env, obj));
        obj.end();

        // --- OGRANICENJA ---

        // Kapacitet fabrika
        for (IloInt i = 0; i < I; i++) {
            IloExpr expr(env);
            for (IloInt j = 0; j < J; j++)
                expr += x[idx_x(i,j)];

            model.add(expr - s[i] <= 0);  // CHANGED: constraint style
            expr.end();
        }

        // Zadovoljavanje potraznje kupaca
        for (IloInt k = 0; k < K; k++) {
            IloExpr expr(env);
            for (IloInt j = 0; j < J; j++)
                expr += y[idx_y(j,k)];

            model.add(expr - d[k] == 0); // CHANGED: constraint style
            expr.end();
        }

        // Konzervacija toka na depou
        for (IloInt j = 0; j < J; j++) {
            IloExpr expr(env);
            for (IloInt i = 0; i < I; i++)
                expr += x[idx_x(i,j)];
            for (IloInt k = 0; k < K; k++)
                expr -= y[idx_y(j,k)];

            model.add(expr == 0);        // CHANGED: single expression
            expr.end();
        }

        // Link x[i][j] i z[i][j]
        for (IloInt i = 0; i < I; i++)
            for (IloInt j = 0; j < J; j++)
                model.add(x[idx_x(i,j)] - s[i] * z[idx_x(i,j)] <= 0); // CHANGED

        // Link y[j][k] i w[j][k]
        for (IloInt j = 0; j < J; j++)
            for (IloInt k = 0; k < K; k++)
                model.add(y[idx_y(j,k)] - d[k] * w[idx_y(j,k)] <= 0); // CHANGED

        // --- SOLVE ---
        IloCplex cplex(model);
        cplex.setParam(IloCplex::TiLim, 7200);
        cplex.setParam(IloCplex::Threads, 1);
        cplex.setOut(env.getNullStream());

        bool solved = cplex.solve();

        std::cout << std::fixed << std::setprecision(2);

        if (solved && cplex.isPrimalFeasible()) {
            std::string statusStr =
                (cplex.getStatus() == IloAlgorithm::Optimal) ? "OPTIMAL" : "FEASIBLE";

            std::cout << argv[1] << " " << statusStr << " "
                      << cplex.getObjValue() << " "
                      << cplex.getTime() << " "
                      << cplex.getNiterations() << " "
                      << cplex.getNnodes() << "\n";
        } else {
            std::cout << argv[1] << " NO_FEASIBLE - "
                      << cplex.getTime() << " "
                      << cplex.getNiterations() << " "
                      << cplex.getNnodes() << "\n";
        }

    } catch (IloException &e) {
        std::cerr << "CPLEX Exception: " << e << std::endl;
    }

    env.end();
    return 0;
}
