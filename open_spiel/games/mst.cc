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
     {"weights", GameParameter(std::string(kWeights))}}
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
  int row = move / num_nodes_;
  int col = move % num_nodes_;

  adjMat_[move] = EdgeState::kConnected;
  adjMat_[col * num_nodes_ + row] = EdgeState::kConnected;

  AddEdge(move); //adds the edge to the adj_list

  reward_ = -weights_[move];
  total_rewards_ += reward_;
  num_moves_ += 1;
}

std::vector<Action> MstState::LegalActions() const {
  if (IsTerminal()) return {};
  // Choose edges where value in adjacency matrix is kAvailable '0'.
  std::vector<Action> moves;
  for (int edge = 0; edge < num_edges_; ++edge) {
    if (adjMat_[edge] == EdgeState::kAvailable && ValidEdge(edge)) {
      moves.push_back(edge);
    }
  }
  return moves;
}

std::string MstState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id % num_nodes_, ",", action_id / num_nodes_, ")");
}

void MstState::AddEdge(int row, int column){ 
  adjList_[row].push_back(column);
  adjList_[column].push_back(row); 
}

void MstState::AddEdge(int edge){
  int row = edge / num_nodes_;
  int col = edge % num_nodes_;
  AddEdge(row, col);
}

/*
void MstState::RemoveEdge(int row, int column) const{
  // Delete column from row in adjList
  std::vector<int>::iterator pos1 = std::find(adjList_[row].begin(), adjList_[row].end(), column);
  if (pos1 != adjList_[row].end()) // == myVector.end() means the element was not found
    adjList_[row].erase(pos1);

  // Delete row from column in adjList
  std::vector<int>::iterator pos2 = std::find(adjList_[column].begin(), adjList_[column].end(), row);
  if (pos2 != adjList_[column].end()) // == myVector.end() means the element was not found
    adjList_[column].erase(pos2);

}

void MstState::RemoveEdge(int edge) const{
  int row = edge / num_nodes_;
  int col = edge % num_nodes_;
  RemoveEdge(row, col);
}
*/
bool MstState::HasNMinus1Edges() const {
  int edgeCount = 0;
  for (int cell = 0; cell < num_edges_; ++cell) {
    if (adjMat_[cell] == EdgeState::kConnected) {
      edgeCount += 1;
    }
  }
  return edgeCount == 2*(num_nodes_ - 1); // matrix is symmetric (so need 2(n-1) edges)
}

bool MstState::IsConnected() const {
  int rowCount = 0;
  for (int r = 0; r < num_nodes_; ++r) {
    for (int c = 0; c < num_nodes_; ++c) {
      if (AdjMatAt(r, c) == EdgeState::kConnected){ //adjMat_[row * num_nodes_ + column]
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
  //std::vector<int>::iterator i; 
  //for (i = adjList_[v].begin(); i != adjList_[v].end(); ++i) 
  for(int i=0; i<adjList_[v].size(); ++i)
  {
    int u = adjList_[v][i];
    // If an adjacent is not visited, then recur for that adjacent 
    if (!visited[u]) 
    { 
    if (IsCyclic(u, visited, v)) 
      return true; 
    } 

    // If an adjacent is visited and not parent of current vertex, 
    // then there is a cycle. 
    else if (u != parent) 
    return true; 
  } 
  return false; 
} 

bool MstState::HasCycle(int r, int c) const{
  bool visited[num_nodes_];
  for (int i = 0; i < num_nodes_; i++) 
      visited[i] = false;
  visited[r] = true;

  if (IsCyclic(c, visited, r)){
    return true; 
  } 
  return false; 
}

bool MstState::ValidEdge(int edge) const{
    int eRow = edge / num_nodes_;
    int eCol = edge % num_nodes_;
    return not HasCycle(eRow, eCol);
}

// Only set the diagonals to kEmpty --> no self-loops
MstState::MstState(std::shared_ptr<const Game> game, int num_nodes, std::vector<float> weights) 
  : State(game),
  num_nodes_(num_nodes),
  weights_(weights),
  num_edges_(num_nodes * num_nodes),
  adjMat_(std::vector<EdgeState>(num_nodes * num_nodes, EdgeState::kAvailable)),
  adjList_(std::vector<std::vector<int>>(num_nodes)) {
  //const MstGame& parent_game = static_cast<const MstGame&>(*game);
  //num_nodes_ = parent_game.NumNodes();
  //weights_ = parent_game.EdgeWeights();

  //num_edges_ = num_nodes_ * num_nodes_;
  //adjMat_ = std::vector<EdgeState>(num_edges_, EdgeState::kAvailable);
  //std::fill(begin(adjMat_), end(adjMat_), EdgeState::kAvailable);
  //adjList_ = new std::vector<int>[num_nodes_];
  
  //Make sure our adjacency list is clean
  for(int i=0; i<num_nodes_;++i){
    adjList_[i].clear();
  }
  for (int r = 0; r < num_nodes_; ++r) {
    //for (int c = 0; c < (r + 1); ++c) {
      //adjMat_[r * num_nodes_ + c] = EdgeState::kAvailable;
    //}
    adjMat_[r * num_nodes_ + r] = EdgeState::kEmpty; // set diagonal to empty
  }
}

std::string MstState::ToString() const {
  std::string str;
  absl::StrAppend(&str, "Num_Nodes: ", num_nodes_, "\n");
  for (int r = 0; r < num_nodes_; ++r) {
    for (int c = 0; c < num_nodes_; ++c) {
      absl::StrAppend(&str, StateToString(AdjMatAt(r, c)));
      absl::StrAppend(&str, ",");
    }
    if (r < (num_nodes_ - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  absl::StrAppend(&str, "\nAdjacency List:\n");
  //std::vector<int>::iterator j;
    for(int i=0; i<num_nodes_;++i){
      absl::StrAppend(&str, i, ": ");
      //for (j = adjList_[i].begin(); j != adjList_[i].end(); ++j)
      for(int j=0; j<adjList_[i].size(); ++j) 
      {
        absl::StrAppend(&str, adjList_[i][j], ",");
      }
      absl::StrAppend(&str, "\n");
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
  TensorView<2> view(values, {kEdgeStates, num_edges_}, true);
  for (int cell = 0; cell < num_edges_; ++cell) {
    view[{static_cast<int>(adjMat_[cell]), cell}] = 1.0;
  }
}

void MstState::UndoAction(Player player, Action move) {
  adjMat_[move] = EdgeState::kAvailable;
  //RemoveEdge(move);
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
}

std::unique_ptr<State> MstState::Clone() const {
  return std::unique_ptr<MstState>(new MstState(*this));
}

MstGame::MstGame(const GameParameters& params)
    : Game(kGameType, params),
      num_nodes_(ParameterValue<int>("num_nodes")),
      edge_weights_(ParseWeights(ParameterValue<std::string>("weights")))
    {}

}  // namespace mst
}  // namespace open_spiel
