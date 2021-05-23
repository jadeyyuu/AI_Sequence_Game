import copy
from template import Agent
import time
import random

from Sequence.sequence_utils import *
from template import GameState, GameRule
from collections import defaultdict
from Sequence.sequence_model import *

class SequenceStateFromValue(SequenceState):
    def __init__(self, state):
        self.deck = state.deck
        self.board = state.board
        self.agents = state.agents
        self.board.draft = state.board.draft

        players_card = None
        for agent_state in self.agents:
            if hasattr(agent_state, 'hand'):
                players_card = agent_state.hand

        cardsfull = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a'] for s in
                     ['d', 'c', 'h', 's']]
        cardsfull = cardsfull * 2

        for i in players_card:
            if i in cardsfull:
                cardsfull.remove(i)

        for i in self.deck.discards:
            if i in cardsfull:
                cardsfull.remove(i)

        for i in self.board.draft:
            if i in cardsfull:
                cardsfull.remove(i)

        self.deck.cards = cardsfull
        for i, agent_state in enumerate(self.agents):
            if not hasattr(agent_state, 'hand'):
                self.agents[i].hand = self.deck.deal(6)


class SequenceGameRuleFromValue(SequenceGameRule):
    def __init__(self, state, agent_index, num_of_agent=4):
        super().__init__(num_of_agent)
        self.perfect_information = True
        self.current_agent_index = agent_index
        self.num_of_agent = num_of_agent
        self.current_game_state = SequenceStateFromValue(state)
        self.action_counter = 0

    def generateSuccessor(self, state, action, agent_id):
        state.board.new_seq = False
        plr_state = state.agents[agent_id]
        plr_state.last_action = action
        reward = 0

        card = action['play_card']
        draft = action['draft_card']
        if card:
            plr_state.hand.remove(card)
            plr_state.discard = card
            state.deck.discards.append(card)
            state.board.draft.remove(draft)
            plr_state.hand.append(draft)
            state.board.draft.extend(state.deck.deal())

        if action['type'] == 'trade':
            reward += 2
            plr_state.trade = True
            return state, 0

        r, c = action['coords']
        if action['type'] == 'place':
            reward += 1
            state.board.chips[r][c] = plr_state.colour
            state.board.empty_coords.remove(action['coords'])
            state.board.plr_coords[plr_state.colour].append(action['coords'])
        elif action['type'] == 'remove':
            reward += 5
            state.board.chips[r][c] = EMPTY
            state.board.empty_coords.append(action['coords'])
        else:
            print("Action unrecognised.")
        if action['type'] == 'place':
            seq, seq_type = self.checkSeq(state.board.chips, plr_state, (r, c))
            if seq:
                reward += seq['num_seq'] * 10
                state.board.new_seq = seq_type
                for sequence in seq['coords']:
                    for r, c in sequence:
                        if state.board.chips[r][c] != JOKER:
                            state.board.chips[r][c] = plr_state.seq_colour
                            try:
                                state.board.plr_coords[plr_state.colour].remove(action['coords'])
                            except:
                                pass
                plr_state.completed_seqs += seq['num_seq']
                plr_state.seq_orientations.extend(seq['orientation'])

        plr_state.trade = False
        plr_state.agent_trace.action_reward.append((action, reward))
        plr_state.score += reward
        return state, reward

    def execute(self, state, action, agent_index):
        temp_state = state
        new_state, reward = self.generateSuccessor(temp_state, action, agent_index)
        current_agent_index = self.getNextAgentIndex() if action['type'] != 'trade' else self.current_agent_index
        return new_state, reward, current_agent_index

    def endState(self, state):
        scores = {RED: 0, BLU: 0}
        for plr_state in state.agents:
            scores[plr_state.colour] += plr_state.completed_seqs
        return scores[RED] >= 2 or scores[BLU] >= 2 or len(state.board.draft) == 0


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.mcts_obj = MCTS(self.id)

    def SelectAction(self, actions, game_state):
        simulator = SequenceGameRuleFromValue(copy.deepcopy(game_state), self.id, 4)
        state = copy.deepcopy(simulator.current_game_state)

        root_node = self.mcts_obj.mcts(simulator, state, self.id, 1)
        return root_node.bestAction()


class Node():
    nextNodeID = 0

    def __init__(self, simulator, parent, state):
        self.simulator = simulator
        self.parent = parent
        self.state = state
        self.id = Node.nextNodeID
        Node.nextNodeID += 1
        self.visits = 0
        self.value = 0.0

    def getValue(self):
        return self.value


class StateNode(Node):
    def __init__(self, parent, state, agent_id, reward=0):
        super().__init__(None, parent, state)
        self.children = {}
        self.reverse_children = {}
        self.reward = reward
        self.agent_id = agent_id

    def isFullyExpanded(self, simulator, state):
        validActions = simulator.getLegalActions(state, self.agent_id)
        validActions = set([self.dictionary_to_tuple(a) for a in validActions])
        if len(validActions) == len(self.children):
            return True
        else:
            return False

    def select(self, simulator, state):
        if not self.isFullyExpanded(simulator, state):
            return self
        else:
            actions = list(self.children.keys())
            qValues = []
            for action in actions:
                qValues.append(self.children[action].getValue())

            # epsilon-greedy
            randomnum = random.random()
            if randomnum < 0.1:
                selected_action = random.choice(actions)
            else:
                maxq = -50000
                best_actions = []
                for action, q in zip(actions, qValues):
                    if q > maxq:
                        maxq = q
                        best_actions = [action]
                    elif q == maxq:
                        best_actions.append(action)
                if len(best_actions) == 0:
                    return None
                selected_action = random.choice(best_actions)
            child = self.children[selected_action]
            return child.select(simulator, child.state)

    def dictionary_to_tuple(self, dict):
        return tuple(dict.items())

    def tuple_to_dictionary(self, tuple):
        return dict(tuple)

    def expand(self, simulator, state, node=None):
        legal_actions = simulator.getLegalActions(state, self.agent_id)
        legal_actions_tuple = set([self.dictionary_to_tuple(a) for a in legal_actions])
        valid_actions = legal_actions_tuple - self.children.keys()
        if len(valid_actions) == 0:
            return None
        action = random.choice(list(valid_actions))
        action = self.tuple_to_dictionary(action)
        new_state, _ = simulator.generateSuccessor(copy.deepcopy(state), action, self.agent_id)
        new_child = StateNode(self, copy.deepcopy(new_state), self.agent_id)
        self.children[self.dictionary_to_tuple(action)] = new_child
        self.reverse_children[new_child] = self.dictionary_to_tuple(action)
        return new_child

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((self.reward + reward - self.value) / self.visits)

        if self.parent != None:
            self.parent.backPropagate(reward * 0.9)  # 0.9 is the discount factor

    def getQFunction(self):
        qValues = {}
        for action in self.children.keys():
            qValues[(self.state, action)] = self.children[action].getValue()
        return qValues

    def bestAction(self):
        """Choose the action with the largest q-value, random tie break."""
        get_q = self.getQFunction()
        maxq = -5000
        best_actions = []
        for (state, action), q in get_q.items():
            if q > maxq:
                maxq = q
                best_actions = [action]
            elif q == maxq:
                best_actions.append(action)
        return self.tuple_to_dictionary(random.choice(best_actions))


class MCTS:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.all_nodes = {}

    def mcts(self, simulator, state, agent_id, timeout=1, print_info=False):
        episode_done = 0
        simulator.current_game_state = copy.deepcopy(state)
        simulator.current_agent_index = agent_id
        rootNode = StateNode(parent=None, state=copy.deepcopy(state), agent_id=agent_id)
        self.all_nodes[rootNode.id] = rootNode

        if print_info:
            print("start mcts")
            print(f"root node: {rootNode.id}")
            self.print_state(rootNode.state)
        if print_info:
            print("loop Selection, Expansion, Simulation, BackPropagation")
        while episode_done < 100:
            if print_info:
                print(f"player: {agent_id}")
                print("present state：")
                self.print_state(simulator.current_game_state)
            if print_info:
                print("selection")
            selectedNode = rootNode.select(simulator, copy.deepcopy(state))
            if selectedNode is None:
                break
            if print_info:
                print(f"selected_node.state:")
                self.print_state(selectedNode.state)
            if print_info:
                print(f"selectednode.id: {selectedNode.id}")
            if selectedNode.id not in self.all_nodes:
                self.all_nodes[selectedNode.id] = selectedNode
            if not simulator.endState(selectedNode.state):
                if print_info:
                    print("expansion")
                child = selectedNode.expand(simulator,copy.deepcopy(selectedNode.state),selectedNode)
                if child.id not in self.all_nodes:
                    self.all_nodes[child.id] = child
                if print_info:
                    print(f"node: {child.id}")
                    print("simulation")
                reward = self.simulate(simulator, child, self.agent_id)
                if print_info:
                    print("backpropagation")
                child.backPropagate(reward)
            else:
                if print_info:
                    print("game over")

            currentTime = int(time.time() * 1000)

            if print_info:
                self.print_tree_node_info()

            episode_done += 1
            if print_info:
                print(f"episode done: {episode_done}")

        if print_info:
            print("done mcts")
            self.print_tree_node_info()

        return rootNode

    def print_tree_node_info(self):
        """Print information of all tree nodes: id, parent id, value, #visit times"""
        for _id, node in self.all_nodes.items():
            if node.parent is None:
                parent_id = -1
            else:
                parent_id = node.parent.id
            print(f"{node.id}, parent id: {parent_id}, value: {node.value}, visits: {node.visits}")

    def print_state(self, state, agent_id=None):
        print("RED")
        print(state.board.plr_coords[RED])
        print("BLUE")
        print(state.board.plr_coords[BLU])
        print("hands")
        for i in range(4):
            print(i, state.agents[i].hand)
        print("draft")
        print(state.board.draft)
        print("board")
        print(BoardToString(state))
        if agent_id is not None:
            print(f"agent {agent_id}.trade: {state.agents[agent_id].trade}")

    def choose(self, simulator, state, agent_id):
        return random.choice(simulator.getLegalActions(state, agent_id))

    def simulate(self, simulator, node, agent_id, print_info=False):
        """Simulate until a terminate state."""
        state = copy.deepcopy(node.state)
        agent_index = agent_id
        if print_info:
            self.print_state(state)
        simulator.current_game_state = state
        simulator.current_agent_index = agent_id
        cumulativeReward = 0.0
        depth = 0

        simulation_episode_done = 0

        while not simulator.endState(state) and not simulator.gameEnds():
            if print_info:
                print(f"no.{simulation_episode_done}simulation")
            if print_info:
                print('\n' + ">>" * 50 + '\n')
                print(f"agent_id{self.agent_id}")
            action = self.choose(simulator, state, agent_index)
            if print_info:
                print(action)
                print(f"excute")
            (state, reward, agent_index) = simulator.execute(state, action, agent_index)
            if print_info:
                print(f"agent_index: {agent_index}")
            simulator.current_agent_index = agent_index
            if print_info:
                self.print_state(state)

            if simulator.endState(state):
                if print_info:
                    print("game over")
                break

            cumulativeReward += pow(0.9, depth) * reward
            depth += 1

            game_end = False
            while agent_index != agent_id:
                if print_info:
                    print(f"agent_index{agent_index}")
                actions = simulator.getLegalActions(state, agent_index)
                selected_action = random.choice(actions)
                if print_info:
                    print(f"selected_action: {selected_action}")
                try:
                    state, _, agent_index = simulator.execute(state, selected_action, agent_index)
                    simulator.current_agent_index = agent_index
                    if print_info:
                        print("situation")
                        self.print_state(state)
                        print(f"agent_index: {agent_index}")
                    if simulator.endState(state):
                        if print_info:
                            print(f"player{agent_index}：game over")
                        game_end = True
                        break
                except:
                    game_end = True

            if game_end:
                break

            simulation_episode_done += 1
        return cumulativeReward
