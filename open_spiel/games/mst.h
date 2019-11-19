// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_OPEN_SPIEL_GAMES_MST_H_
#define THIRD_PARTY_OPEN_SPIEL_GAMES_MST_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Parameters: numNodes: 5-100

namespace open_spiel {
namespace mst {

// Constants.
inline constexpr int kNumPlayers = 1;
inline constexpr int kNumNodes = 30;
inline constexpr int kNumEdges = kNumNodes * kNumNodes; // parameter: number of edges, doesn't have to be fully connected
inline constexpr int kEdgeStates = 3;  // -1, 0, 1 not able to connect, able to connect, connected

// inline constexpr int kNumberStates = 5478;

// State of a cell.
enum class EdgeState {
  kEmpty, // can't choose this edge ever
  kAvailable, // edge can be chosen
  kConnected, // edge is connected
};

// State of an in-play game.
class MstState : public State {
 public:
  MstState(std::shared_ptr<const Game> game);

  MstState(const MstState&) = default;
  MstState& operator=(const MstState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::vector<double> Rewards() const override;
  std::string InformationState(Player player) const override;
  std::string Observation(Player player) const override;
  void ObservationAsNormalizedVector(
      Player player, std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  EdgeState AdjMatAt(int cell) const { return adjMat_[cell]; }
  EdgeState AdjMatAt(int row, int column) const {
    return adjMat_[row * kNumNodes + column];
  }

 protected:
  std::array<EdgeState, kNumEdges> adjMat_;
  std::vector<int> *adjList_;

  void DoApplyAction(Action move) override;

 private:
  bool ValidEdge(int edge) const;
  bool HasCycle(int r, int c) const;
  bool IsCyclic(int v, bool visited[], int parent) const;
  void AddEdge(int row, int column) const;
  void AddEdge(int edge) const;
  bool HasNMinus1Edges() const; // Does the graph have N-1 edges?
  bool IsConnected() const; // Is the graph connected (is there an edge in each row of adjMat)
  Player current_player_ = 1; // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
  // Fields sets to bad/invalid values. Use Game::NewInitialState().
  double total_rewards_ = -1;
  int horizon_ = -1;        // Limit on the total number of moves.
  bool win_;        // True if agents push the big box to the goal.

  // Most recent rewards.
  double reward_;

};

// Game object.
class MstGame : public Game {
 public:
  explicit MstGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumEdges; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new MstState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new MstGame(*this));
  }
  std::vector<int> ObservationNormalizedVectorShape() const override {
    return {kEdgeStates, kNumNodes, kNumNodes};
  }
  int MaxGameLength() const { return kNumEdges; }
};

EdgeState PlayerToState(Player player);
std::string StateToString(EdgeState state);

inline std::ostream& operator<<(std::ostream& stream, const EdgeState& state) {
  return stream << StateToString(state);
}

}  // namespace mst
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_MST_H_
