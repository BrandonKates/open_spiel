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

#include "open_spiel/games/mst.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <list>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/tensor_view.h"

namespace open_spiel {
namespace mst {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"mst",
    /*long_name=*/"Minimum Spanning Tree",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kIdentical,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/1,
    /*min_num_players=*/1,
    /*provides_information_state=*/true,
    /*provides_information_state_as_normalized_vector=*/false,
    /*provides_observation=*/true,
    /*provides_observation_as_normalized_vector=*/true,
    /*parameter_specification=*/
    {{"num_nodes", GameParameter(kNumNodes)},
     {"weights", GameParameter(kWeights)}}
   };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new MstGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

EdgeState PlayerToState(Player player) {
  switch (player) {
    case -1:
      return EdgeState::kEmpty;
    case 0:
      return EdgeState::kAvailable;
    case 1:
      return EdgeState::kConnected;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return EdgeState::kEmpty;
  }
}

std::string StateToString(EdgeState state) {
  switch (state) {
    case EdgeState::kEmpty:
      return "0";
    case EdgeState::kAvailable:
      return "0";
    case EdgeState::kConnected:
      return "1";
    default:
      SpielFatalError("Unknown state.");
  }
}

std::vector<float> ParseWeights(std::string values){
  std::stringstream ss(values);
  std::vector<float> result;

  while( ss.good() )
  {
    std::string substr;
    getline( ss, substr, ',' );
    result.push_back( std::stof(substr) );
  }
  return result;
}

void MstState::DoApplyAction(Action move) {
  SPIEL_CHECK_EQ(adjMat_[move], EdgeState::kAvailable);
  int row = move / kNumNodes;
  int col = move % kNumNodes;

  adjMat_[move] = EdgeState::kConnected;
  adjMat_[col * kNumNodes + row] = EdgeState::kConnected;
  AddEdge(move); //adds the edge to the adj_list

  reward_ = -weights_[move];
  total_rewards_ += reward_;
  num_moves_ += 1;
}

std::vector<Action> MstState::LegalActions() const {
  if (IsTerminal()) return {};
  // Choose edges where value in adjacency matrix is kAvailable '0'.
  std::vector<Action> moves;
  /*for (int r=0; r<kNumNodes; ++r){
    for(int c=0; c<kNumNodes; ++c){
      int edge = r * kNumNodes + c;
      int otherEdge = c * kNumNodes + r;

    }
  } */
  for (int edge = 0; edge < kNumEdges; ++edge) {
    if (adjMat_[edge] == EdgeState::kAvailable && ValidEdge(edge)) {
      moves.push_back(edge);
    }
  }
  return moves;
}

std::string MstState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id % kNumNodes, ",", action_id / kNumNodes, ")");
}

void MstState::AddEdge(int row, int column) const{ 
  adjList_[row].push_back(column);
  adjList_[column].push_back(row); 
}

void MstState::AddEdge(int edge) const{
  int row = edge / kNumNodes;
  int col = edge % kNumNodes;
  AddEdge(row, col);
}

bool MstState::HasNMinus1Edges() const {
  int edgeCount = 0;
  for (int cell = 0; cell < kNumEdges; ++cell) {
    if (adjMat_[cell] == EdgeState::kConnected) {
      edgeCount += 1;
    }
  }
  return edgeCount == 2*(kNumNodes - 1); // matrix is symmetric (so need 2(n-1) edges)
}

bool MstState::IsConnected() const {
  int rowCount = 0;
  for (int r = 0; r < kNumNodes; ++r) {
    for (int c = 0; c < kNumNodes; ++c) {
      if (AdjMatAt(r, c) == EdgeState::kConnected){ //adjMat_[row * kNumNodes + column]
        rowCount += 1;
        break;
      }
    }
    if (rowCount == 0){
      return false;
    }
    rowCount = 0;
  }
  return true;
}

bool MstState::IsCyclic(int v, bool visited[], int parent) const{
  // Mark the current node as visited 
  visited[v] = true; 

  // Recur for all the vertices adjacent to this vertex 
  std::vector<int>::iterator i; 
  for (i = adjList_[v].begin(); i != adjList_[v].end(); ++i) 
  { 
    // If an adjacent is not visited, then recur for that adjacent 
    if (!visited[*i]) 
    { 
    if (IsCyclic(*i, visited, v)) 
      return true; 
    } 

    // If an adjacent is visited and not parent of current vertex, 
    // then there is a cycle. 
    else if (*i != parent) 
    return true; 
  } 
  return false; 
} 

bool MstState::HasCycle(int r, int c) const{
  bool *visited = new bool[kNumNodes]; 
  for (int i = 0; i < kNumNodes; i++) 
      visited[i] = false;
  visited[r] = true;

  if (IsCyclic(c, visited, r)){
    //std::cout << c << "," << r << " contains cycle.\n";
    return true; 
  } 
  return false; 
}

bool MstState::ValidEdge(int edge) const{
    int eRow = edge / kNumNodes;
    int eCol = edge % kNumNodes;
    return not HasCycle(eRow, eCol);
}

// Only set the diagonals to kEmpty --> no self-loops
MstState::MstState(std::shared_ptr<const Game> game) : State(game) {
  const MstGame& parent_game = static_cast<const MstGame&>(*game);
  int num_nodes_ = parent_game.NumNodes();
  //std::string edge_weights_ = parent_game.EdgeWeights();
  weights_ = parent_game.EdgeWeights();//ParseWeights(edge_weights_);
  // Process edge_weights and set num_nodes_ to the number of nodes in the game

  std::fill(begin(adjMat_), end(adjMat_), EdgeState::kAvailable);
  adjList_ = new std::vector<int>[kNumNodes];
  for (int r = 0; r < kNumNodes; ++r) {
    //for (int c = 0; c < (r + 1); ++c) {
      //adjMat_[r * kNumNodes + c] = EdgeState::kAvailable;
    //}
    adjMat_[r * kNumNodes + r] = EdgeState::kEmpty; // set diagonal to empty
  }
}

std::string MstState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumNodes; ++r) {
    for (int c = 0; c < kNumNodes; ++c) {
      absl::StrAppend(&str, StateToString(AdjMatAt(r, c)));
      absl::StrAppend(&str, ",");
    }
    if (r < (kNumNodes - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool MstState::IsTerminal() const {
  //return outcome_ != kInvalidPlayer || HasNMinus1Edges();
  return outcome_ != kInvalidPlayer || HasNMinus1Edges();// && IsConnected());
}

std::string MstState::InformationState(Player player) const {
  return HistoryString();
}

std::string MstState::Observation(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void MstState::ObservationAsNormalizedVector(
    Player player, std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kEdgeStates, kNumEdges}, true);
  for (int cell = 0; cell < kNumEdges; ++cell) {
    view[{static_cast<int>(adjMat_[cell]), cell}] = 1.0;
  }
}

void MstState::UndoAction(Player player, Action move) {
  adjMat_[move] = EdgeState::kAvailable;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
}

std::unique_ptr<State> MstState::Clone() const {
  return std::unique_ptr<State>(new MstState(*this));
}

MstGame::MstGame(const GameParameters& params)
    : Game(kGameType, params),
      num_nodes_(ParameterValue<int>("num_nodes")),
      edge_weights_(ParseWeights(ParameterValue<std::string>("weights")))
    {}

}  // namespace mst
}  // namespace open_spiel
