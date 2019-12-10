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
inline constexpr int kNumNodes = 1;
inline constexpr int kEdgeStates = 3;  // -1, 0, 1 not able to connect, able to connect, connected
inline constexpr auto kWeights = "0";//"0.0,0.33,0.37,0.19,0.84,0.33,0.0,0.18,0.42,0.58,0.37,0.18,0.0,0.39,0.46,0.19,0.42,0.39,0.0,0.82,0.84,0.58,0.46,0.82,0.0";

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
  MstState(std::shared_ptr<const Game>, int num_nodes, std::vector<float> weights);


  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : current_player_;
  }
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override { std::vector<double> temp_ = {total_rewards_}; return temp_;};
  std::vector<double> Rewards() const override { std::vector<double> temp_ = {reward_}; return temp_;};
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(
      Player player, std::vector<double>* values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  EdgeState AdjMatAt(int cell) const { return adjMat_[cell]; }
  EdgeState AdjMatAt(int row, int column) const {
    return adjMat_[row * num_nodes_ + column];
  }

 protected:
  int num_nodes_ = kNumNodes;
  int num_edges_ = kNumNodes * kNumNodes;
  std::vector<EdgeState> adjMat_;
  std::vector<float> weights_;
  std::vector<std::vector<int>> adjList_;

  void DoApplyAction(Action move) override;

 private:
  bool ValidEdge(int edge) const;
  bool HasCycle(int r, int c) const;
  bool IsCyclic(int v, bool visited[], int parent) const;
  void AddEdge(int row, int column);
  void AddEdge(int edge);
  /*
  void RemoveEdge(int row, int column) const;
  void RemoveEdge(int edge) const;
  */
  bool HasNMinus1Edges() const; // Does the graph have N-1 edges?
  bool IsConnected() const; // Is the graph connected (is there an edge in each row of adjMat)
  Player current_player_ = 0; // Player zero goes first
  Player outcome_ = kInvalidPlayer;
  int num_moves_ = 0;
  // Fields sets to bad/invalid values. Use Game::NewInitialState().

  // Most recent rewards.
  double reward_ = 0;
  double total_rewards_ = 0;

};

// Game object.
class MstGame : public Game {
 public:
  explicit MstGame(const GameParameters& params);
  int NumDistinctActions() const override { return num_nodes_ * num_nodes_; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(
        new MstState(shared_from_this(), num_nodes_, edge_weights_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::shared_ptr<const Game> Clone() const override {
    return std::shared_ptr<const Game>(new MstGame(*this));
  }
  std::vector<int> ObservationTensorShape() const override {
    return {kEdgeStates, kNumNodes, kNumNodes};
  }
  int MaxGameLength() const { return num_nodes_ * num_nodes_; }
  int NumNodes() const { return num_nodes_; }
  std::vector<float> EdgeWeights() const { return edge_weights_; }

 private:
  int num_nodes_ = 0; //set some defaults
  std::vector<float> edge_weights_ = {};
};

EdgeState PlayerToState(Player player);
std::string StateToString(EdgeState state);
std::vector<float> ParseWeights(std::string values);

inline std::ostream& operator<<(std::ostream& stream, const EdgeState& state) {
  return stream << StateToString(state);
}

}  // namespace mst
}  // namespace open_spiel

#endif  // THIRD_PARTY_OPEN_SPIEL_GAMES_MST_H_
