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

    IloEnv env;   // 🔧 FIX: env MUST be created before any Concert container

    // --- UCITAVANJE INSTANCE ---
    IloInt I, J, K;
    in >> I >> J >> K;

    IloNumArray s(env, I);   // 🔧 FIX: pass env
    IloNumArray d(env, K);   // 🔧 FIX: pass env

    for (IloInt i = 0; i < I; i++) in >> s[i];
    for (IloInt k = 0; k < K; k++) in >> d[k];

    IloArray<IloNumArray> b(env, I);   // 🔧 FIX: pass env
    IloArray<IloNumArray> f(env, I);   // 🔧 FIX: pass env
    IloArray<IloNumArray> c(env, J);   // 🔧 FIX: pass env
    IloArray<IloNumArray> g(env, J);   // 🔧 FIX: pass env

    for (IloInt i = 0; i < I; i++) {
        b[i] = IloNumArray(env, J);   // 🔧 FIX: pass env
        f[i] = IloNumArray(env, J);   // 🔧 FIX: pass env
    }

    for (IloInt j = 0; j < J; j++) {
        c[j] = IloNumArray(env, K);   // 🔧 FIX: pass env
        g[j] = IloNumArray(env, K);   // 🔧 FIX: pass env
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

    try {
        IloModel model(env);

        // --- VARIJABLE ---
        IloNumVarArray x(env, I * J, 0.0, IloInfinity, ILOFLOAT);
        IloBoolVarArray z(env, I * J);
        IloNumVarArray y(env, J * K, 0.0, IloInfinity, ILOFLOAT);
        IloBoolVarArray w(env, J * K);

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

        for (IloInt i = 0; i < I; i++) {
            IloExpr expr(env);
            for (IloInt j = 0; j < J; j++)
                expr += x[idx_x(i,j)];
            model.add(expr - s[i] <= 0);
            expr.end();
        }

        for (IloInt k = 0; k < K; k++) {
            IloExpr expr(env);
            for (IloInt j = 0; j < J; j++)
                expr += y[idx_y(j,k)];
            model.add(expr - d[k] == 0);
            expr.end();
        }

        for (IloInt j = 0; j < J; j++) {
            IloExpr expr(env);
            for (IloInt i = 0; i < I; i++)
                expr += x[idx_x(i,j)];
            for (IloInt k = 0; k < K; k++)
                expr -= y[idx_y(j,k)];
            model.add(expr == 0);
            expr.end();
        }

        for (IloInt i = 0; i < I; i++)
            for (IloInt j = 0; j < J; j++)
                model.add(x[idx_x(i,j)] - s[i] * z[idx_x(i,j)] <= 0);

        for (IloInt j = 0; j < J; j++)
            for (IloInt k = 0; k < K; k++)
                model.add(y[idx_y(j,k)] - d[k] * w[idx_y(j,k)] <= 0);

        // --- SOLVE ---
        IloCplex cplex(model);

        // 🔧 OPTIONAL FIX (avoids deprecation warnings)
        cplex.setParam(IloCplex::Param::TimeLimit, 7200);
        cplex.setParam(IloCplex::Param::Threads, 1);

        cplex.setOut(env.getNullStream());

        if (cplex.solve() && cplex.isPrimalFeasible()) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << argv[1] << " "
                      << ((cplex.getStatus() == IloAlgorithm::Optimal) ? "OPTIMAL" : "FEASIBLE") << " "
                      << cplex.getObjValue() << " "
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
