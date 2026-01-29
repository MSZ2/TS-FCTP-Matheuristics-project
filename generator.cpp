#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

struct Node {
  int x, y;
};

int dist(const Node &a, const Node &b) {
  return (int)std::floor(
      std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)));
}

void generate_group(const std::string &prefix,
                    const std::vector<std::vector<int>> &dims,
                    std::mt19937 &rng) {

  std::uniform_int_distribution<int> coord(-400, 400);
  std::uniform_int_distribution<int> rf(10, 15);
  std::uniform_int_distribution<int> rg(5, 10);
  std::uniform_int_distribution<int> dem(10, 30);

  for (int t = 0; t < (int)dims.size(); t++) {
    int I = dims[t][0];
    int J = dims[t][1];
    int K = dims[t][2];

    std::vector<Node> P(I), D(J), C(K);
    for (auto &x : P)
      x = {coord(rng), coord(rng)};
    for (auto &x : D)
      x = {coord(rng), coord(rng)};
    for (auto &x : C)
      x = {coord(rng), coord(rng)};

    std::vector<int> s(I, 80);
    std::vector<int> d(K);
    int supply = I * 80, demand = 0;

    for (int k = 0; k < K; k++) {
      d[k] = dem(rng);
      demand += d[k];
    }

    for (int k = 0; demand != supply; k = (k + 1) % K) {
      if (demand > supply && d[k] > 1) {
        d[k]--;
        demand--;
      } else if (demand < supply) {
        d[k]++;
        demand++;
      }
    }

    std::ofstream out("generated_instances/" + prefix + "_" +
                      std::to_string(t + 1) + ".txt");

    out << I << " " << J << " " << K << "\n";

    for (int i = 0; i < I; i++)
      out << s[i] << " ";
    out << "\n";

    for (int k = 0; k < K; k++)
      out << d[k] << " ";
    out << "\n";

    for (int i = 0; i < I; i++) {
      for (int j = 0; j < J; j++)
        out << dist(P[i], D[j]) << " ";
      out << "\n";
    }

    for (int i = 0; i < I; i++) {
      int r = rf(rng);
      for (int j = 0; j < J; j++)
        out << dist(P[i], D[j]) * r << " ";
      out << "\n";
    }

    for (int j = 0; j < J; j++) {
      for (int k = 0; k < K; k++)
        out << dist(D[j], C[k]) << " ";
      out << "\n";
    }

    for (int j = 0; j < J; j++) {
      int r = rg(rng);
      for (int k = 0; k < K; k++)
        out << dist(D[j], C[k]) * r << " ";
      out << "\n";
    }

    out.close();
  }
}

int main() {

  std::mt19937 rng(1389);

  // ---------- MALE INSTANCE ----------
  // I, J, K mali
  std::vector<std::vector<int>> small = {
      {4, 8, 16},   {4, 8, 20},   {4, 10, 20},  {6, 12, 24},  {6, 12, 30},
      {6, 15, 30},  {8, 16, 32},  {8, 16, 40},  {8, 20, 40},  {10, 20, 40},
      {10, 20, 50}, {10, 25, 50}, {12, 24, 48}, {12, 24, 60}, {12, 30, 60},
      {14, 28, 56}, {14, 28, 70}, {16, 32, 64}, {16, 32, 80}, {18, 36, 72}};

  // ---------- SREDNJE INSTANCE ----------
  std::vector<std::vector<int>> medium = {
      {20, 40, 80},   {20, 40, 100},  {20, 50, 100},  {25, 50, 100},
      {25, 50, 125},  {25, 60, 120},  {30, 60, 120},  {30, 60, 150},
      {30, 75, 150},  {35, 70, 140},  {35, 70, 175},  {40, 80, 160},
      {40, 80, 200},  {45, 90, 180},  {45, 90, 225},  {50, 100, 200},
      {50, 100, 250}, {55, 110, 220}, {55, 110, 275}, {60, 120, 240}};

  // ---------- VELIKE INSTANCE ----------
  std::vector<std::vector<int>> large = {
      {70, 140, 280},  {70, 140, 350},  {80, 160, 320},  {80, 160, 400},
      {90, 180, 360},  {90, 180, 450},  {100, 200, 400}, {100, 200, 500},
      {110, 220, 440}, {110, 220, 550}, {120, 240, 480}, {120, 240, 600},
      {130, 260, 520}, {130, 260, 650}, {140, 280, 560}, {140, 280, 700},
      {150, 300, 600}, {150, 300, 750}, {160, 320, 640}, {160, 320, 800}};
  std::vector<std::vector<int>> tiny = {{3, 5, 10}, {4, 6, 12}, {5, 8, 15}};
  std::vector<std::vector<int>> baseline = {
      {10, 20, 40}, {15, 25, 50}, {20, 30, 60}, {25, 40, 80}};

  generate_group("tiny", tiny, rng);
  generate_group("baseline", baseline, rng);
  generate_group("small", small, rng);
  generate_group("medium", medium, rng);
  generate_group("large", large, rng);

  return 0;
}
