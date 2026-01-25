#include <fstream>
#include <ilcplex/ilocplex.h>
#include <iostream>
#include <iomanip>
#include <vector>

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
    int I, J, K; // Broj fabrika, depoa, kupaca
    in >> I >> J >> K;

    std::vector<double> s(I); // kapacitet fabrika
    std::vector<double> d(K); // potraznja kupaca
    for (int i = 0; i < I; i++) in >> s[i];
    for (int k = 0; k < K; k++) in >> d[k];

    // Troškovi transporta i fiksni troškovi
    std::vector<std::vector<double>> b(I, std::vector<double>(J));
    std::vector<std::vector<double>> f(I, std::vector<double>(J));
    std::vector<std::vector<double>> c(J, std::vector<double>(K));
    std::vector<std::vector<double>> g(J, std::vector<double>(K));

    for (int i = 0; i < I; i++)
        for (int j = 0; j < J; j++) in >> b[i][j];
    for (int i = 0; i < I; i++)
        for (int j = 0; j < J; j++) in >> f[i][j];
    for (int j = 0; j < J; j++)
        for (int k = 0; k < K; k++) in >> c[j][k];
    for (int j = 0; j < J; j++)
        for (int k = 0; k < K; k++) in >> g[j][k];

    in.close();

    IloEnv env;
    try {
        IloModel model(env);

        // --- VARIJABLE ---
        // Flatten 2D varijable x[i][j] u 1D za bolje performanse
        IloNumVarArray x(env, I * J, 0.0, IloInfinity, ILOFLOAT);
        IloNumVarArray z(env, I * J, 0.0, 1.0, ILOBOOL);
        IloNumVarArray y(env, J * K, 0.0, IloInfinity, ILOFLOAT);
        IloNumVarArray w(env, J * K, 0.0, 1.0, ILOBOOL);

        auto idx_x = [J](int i, int j) { return i * J + j; };
        auto idx_y = [K](int j, int k) { return j * K + k; };

        // --- OBJEKTIVNA FUNKCIJA ---
        IloExpr obj(env);
        for (int i = 0; i < I; i++)
            for (int j = 0; j < J; j++)
                obj += b[i][j] * x[idx_x(i, j)] + f[i][j] * z[idx_x(i, j)];

        for (int j = 0; j < J; j++)
            for (int k = 0; k < K; k++)
                obj += c[j][k] * y[idx_y(j, k)] + g[j][k] * w[idx_y(j, k)];

        model.add(IloMinimize(env, obj));
        obj.end();

        // --- OGRANICENJA ---

        // Kapacitet fabrika
        for (int i = 0; i < I; i++) {
            IloExpr expr(env);
            for (int j = 0; j < J; j++)
                expr += x[idx_x(i, j)];
            model.add(expr <= s[i]);
            expr.end();
        }

        // Zadovoljavanje potraznje kupaca
        for (int k = 0; k < K; k++) {
            IloExpr expr(env);
            for (int j = 0; j < J; j++)
                expr += y[idx_y(j, k)];
            model.add(expr == d[k]);
            expr.end();
        }

        // Konzervacija toka na depou (uključuje sve dolazne i odlazne tokove)
        for (int j = 0; j < J; j++) {
            IloExpr in_expr(env);
            IloExpr out_expr(env);
            for (int i = 0; i < I; i++)
                in_expr += x[idx_x(i, j)];
            for (int k = 0; k < K; k++)
                out_expr += y[idx_y(j, k)];
            model.add(in_expr == out_expr);
            in_expr.end();
            out_expr.end();
        }

        // Link x[i][j] i z[i][j] (ako z=0, x=0)
        for (int i = 0; i < I; i++)
            for (int j = 0; j < J; j++)
                model.add(x[idx_x(i, j)] <= s[i] * z[idx_x(i, j)]);

        // Link y[j][k] i w[j][k] (ako w=0, y=0)
        for (int j = 0; j < J; j++)
            for (int k = 0; k < K; k++)
                model.add(y[idx_y(j, k)] <= d[k] * w[idx_y(j, k)]);

        // --- SOLVE ---
        IloCplex cplex(model);

        cplex.setParam(IloCplex::TiLim, 7200); // 2h limit
        cplex.setParam(IloCplex::Threads, 1);
        cplex.setOut(env.getNullStream()); // ispis isključen

        bool solved = cplex.solve();

        std::cout << std::fixed << std::setprecision(2);

        if (solved) {
            IloAlgorithm::Status status = cplex.getStatus();
            double time = cplex.getTime();
            double iters = cplex.getNiterations();
            double nodes = cplex.getNnodes();

            if (cplex.isPrimalFeasible()) {
                double objVal = cplex.getObjValue();
                std::string statusStr = (status == IloAlgorithm::Optimal) ? "OPTIMAL" : "FEASIBLE";

                std::cout << argv[1] << " " << statusStr << " "
                          << objVal << " " << time << " " << iters << " " << nodes << "\n";
            } else {
                std::cout << argv[1] << " NO_FEASIBLE - " << time << " "
                          << iters << " " << nodes << "\n";
            }
        } else {
            std::cout << argv[1] << " NO_FEASIBLE - "
                      << cplex.getTime() << " " << cplex.getNiterations()
                      << " " << cplex.getNnodes() << "\n";
        }

    } catch (IloException &e) {
        std::cerr << "CPLEX Exception: " << e << std::endl;
    }

    env.end();
    return 0;
}
