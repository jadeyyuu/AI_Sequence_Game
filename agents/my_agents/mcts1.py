import copy
from template import Agent
import time
import random



from Sequence.sequence_utils import *
from template import GameState, GameRule
from collections import defaultdict
from Sequence.sequence_model import *

#from agents.my_agents.utils import *
class SequenceStateFromValue(SequenceState):
    def __init__(self, state):
        # 此处我们拿到的state是不完整的，缺少
        # * 每个玩家手里的牌
        # * deck中剩余的牌(扣着的牌)
        self.deck = state.deck
        self.board = state.board
        self.agents = state.agents
        self.board.draft = state.board.draft

        """通过给其他3个玩家随机发放6张牌，创建一个'虚拟'的游戏状态，从而可以进行模拟"""

        # 当前玩家手中的牌 (其他玩家手中的牌已被屏蔽，无法获取)
        players_card = None
        for agent_state in self.agents:
            if hasattr(agent_state, 'hand'):
                players_card = agent_state.hand
        # 全部的牌
        cardsfull = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a'] for s in
                     ['d', 'c', 'h', 's']]
        cardsfull = cardsfull * 2  # Sequence uses 2 decks.

        # 去掉当前玩家手中的牌
        for i in players_card:
            if i in cardsfull:
                cardsfull.remove(i)

        # 去掉整个游戏过程中已经丢弃的牌
        for i in self.deck.discards:
            if i in cardsfull:
                cardsfull.remove(i)

        # 去掉公共牌池的牌
        for i in self.board.draft:
            if i in cardsfull:
                cardsfull.remove(i)

        # 剩下的牌池(与真实的基本上完全一致)
        self.deck.cards = cardsfull

        # 从剩下的牌中随机给其他3位玩家每人发6张牌
        for i, agent_state in enumerate(self.agents):
            if not hasattr(agent_state, 'hand'):
                self.agents[i].hand = self.deck.deal(6)


class SequenceGameRuleFromValue(SequenceGameRule):
    """Game rule that can be created from a specified game state and agent index."""

    def __init__(self, state, agent_index, num_of_agent=4):
        super().__init__(num_of_agent)
        self.perfect_information = True
        self.current_agent_index = agent_index
        self.num_of_agent = num_of_agent
        self.current_game_state = SequenceStateFromValue(state)  # 使用我们手动创建的"虚拟游戏状态"
        self.action_counter = 0

    def checkSeq(self, chips, plr_state, last_coords):
        clr,sclr   = plr_state.colour, plr_state.seq_colour
        oc,os      = plr_state.opp_colour, plr_state.opp_seq_colour
        seq_type   = TRADSEQ
        seq_coords = []
        seq_found  = {'vr':0, 'hz':0, 'd1':0, 'd2':0, 'hb':0}
        found      = False
        nine_chip  = lambda x,clr : len(x)==9 and len(set(x))==1 and clr in x
        lr,lc      = last_coords
        
        #All joker spaces become player chips for the purposes of sequence checking.
        for r,c in COORDS['jk']:
            chips[r][c] = clr
        
        #First, check "heart of the board" (2h, 3h, 4h, 5h). If possessed by one team, the game is over.
        coord_list = [(4,4),(4,5),(5,4),(5,5)]
        heart_chips = [chips[y][x] for x,y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (oc in heart_chips or os in heart_chips):
            seq_type = HOTBSEQ
            seq_found['hb']+=2
            seq_coords.append(coord_list)
            
        #Search vertical, horizontal, and both diagonals.
        vr = [(-4,0),(-3,0),(-2,0),(-1,0),(0,0),(1,0),(2,0),(3,0),(4,0)]
        hz = [(0,-4),(0,-3),(0,-2),(0,-1),(0,0),(0,1),(0,2),(0,3),(0,4)]
        d1 = [(-4,-4),(-3,-3),(-2,-2),(-1,-1),(0,0),(1,1),(2,2),(3,3),(4,4)]
        d2 = [(-4,4),(-3,3),(-2,2),(-1,1),(0,0),(1,-1),(2,-2),(3,-3),(4,-4)]
        for seq,seq_name in [(vr,'vr'), (hz,'hz'), (d1,'d1'), (d2,'d2')]:
            coord_list = [(r+lr, c+lc) for r,c in seq]
            coord_list = [i for i in coord_list if 0<=min(i) and 9>=max(i)] #Sequences must stay on the board.
            chip_str   = ''.join([chips[r][c] for r,c in coord_list])
            #Check if there exists 4 player chips either side of new chip (counts as forming 2 sequences).
            if nine_chip(chip_str, clr):
                seq_found[seq_name]+=2
                seq_coords.append(coord_list)
            #If this potential sequence doesn't overlap an established sequence, do fast check.
            if sclr not in chip_str:
                sequence_len = 0
                start_idx    = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == clr:
                        sequence_len += 1
                    else:
                        start_idx = i+1
                        sequence_len = 0
                    if sequence_len >= 5:
                        seq_found[seq_name] += 1
                        seq_coords.append(coord_list[start_idx:start_idx+5])    
                        break
            else: #Check for sequences of 5 player chips, with a max. 1 chip from an existing sequence.
                for pattern in [clr*5, clr*4+sclr, clr*3+sclr+clr, clr*2+sclr+clr*2, clr+sclr+clr*3, sclr+clr*4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx+5] == pattern:
                            seq_found[seq_name]+=1
                            seq_coords.append(coord_list[start_idx:start_idx+5])
                            found = True
                            break
                    if found:
                        break
        
        for r,c in COORDS['jk']:
            chips[r][c] = JOKER #Joker spaces reset after sequence checking.
        
        num_seq = sum(seq_found.values())
        if num_seq > 1 and seq_type != HOTBSEQ:
            seq_type = MULTSEQ
        return ({'num_seq':num_seq, 'orientation':[k for k,v in seq_found.items() if v], 'coords':seq_coords}, seq_type) if num_seq else (None,None)

    def generateSuccessor(self, state, action, agent_id):
        state.board.new_seq = False
        # print(f"agent id {agent_id}")
        plr_state = state.agents[agent_id]
        plr_state.last_action = action  # Record last action such that other agents can make use of this information.
        reward = 0

        # Update agent state. Take the card in play from the agent, discard, draw the selected draft, deal a new draft.
        # If agent was allowed to trade but chose not to, there is no card played, and hand remains the same.
        # print(action)
        #更新棋盘状态（手牌，弃牌堆，牌库，展示台）
        card = action['play_card']
        draft = action['draft_card']
        if card:
            plr_state.hand.remove(card)  # Remove card from hand.
            plr_state.discard = card  # Add card to discard pile.
            state.deck.discards.append(
                card)  # Add card to global list of discards (some agents might find tracking this helpful).
            state.board.draft.remove(draft)  # Remove draft from draft selection.
            plr_state.hand.append(draft)  # Add draft to player hand.
            state.board.draft.extend(state.deck.deal())  # Replenish draft selection.

        # If action was to trade in a dead card, action is complete, and agent gets to play another card.
        # reward可以在此处增加(如发现展示台有需要的牌)
        if action['type'] == 'trade':
            plr_state.trade = True  # Switch trade flag to prohibit agent performing a second trade this turn.
            return state, 0

        # Update Sequence board. If action was to place/remove a marker, add/subtract it from the board.
        r, c = action['coords']
        if action['type'] == 'place':
            state.board.chips[r][c] = plr_state.colour
            state.board.empty_coords.remove(action['coords'])
            state.board.plr_coords[plr_state.colour].append(action['coords'])
        elif action['type'] == 'remove':
            state.board.chips[r][c] = EMPTY
            state.board.empty_coords.append(action['coords'])
        else:
            print("Action unrecognised.")

        # Check if a sequence has just been completed. If so, upgrade chips to special sequence chips.
        if action['type'] == 'place':
            seq, seq_type = self.checkSeq(state.board.chips, plr_state, (r, c))
            if seq:
                reward += seq['num_seq'] * 10000
                state.board.new_seq = seq_type
                for sequence in seq['coords']:
                    for r, c in sequence:
                        if state.board.chips[r][c] != JOKER:  # Joker spaces stay jokers.
                            state.board.chips[r][c] = plr_state.seq_colour
                            try:
                                state.board.plr_coords[plr_state.colour].remove(action['coords'])
                            except:  # Chip coords were already removed with the first sequence.
                                pass
                plr_state.completed_seqs += seq['num_seq']
                plr_state.seq_orientations.extend(seq['orientation'])
            reward += self.getReward(state, action, agent_id)
        plr_state.trade = False  # Reset trade flag if agent has completed a full turn.
        plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and any resultant score.
        plr_state.score += reward
        return state, reward


    def getReward(self, state, action, agent_id):
        plr_state = state.agents[agent_id]
        value = 100

        #get start position
        r,c = action['coords']
        #print("action:", action)
        start = (int(r), int(c))\
        #get all self chips
        for r in range(10):
            for c in range(10):
                if state.board.chips[r][c] == plr_state.colour:
                    destination = (int(r), int(c))
                    value -= self.manhattan(start, destination)
        return value
    #get manhattan distance from two point
    def manhattan(self, start, destination):  
        xy1 = start
        #print("start:", start)
        xy2 = destination
        #print("destination:", destination)
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    def execute(self, state, action, agent_index):
        temp_state = state
        new_state, reward = self.generateSuccessor(temp_state, action, agent_index)
        current_agent_index = self.getNextAgentIndex() if action['type'] != 'trade' else self.current_agent_index
        return new_state, reward, current_agent_index

    def getNextAgentIndex(self):
        return (self.current_agent_index + 1) % self.num_of_agent

    def endState(self, state):
        """检测是否是游戏终止状态"""
        scores = {RED: 0, BLU: 0}
        for plr_state in state.agents:
            scores[plr_state.colour] += plr_state.completed_seqs
        return scores[RED] >= 2 or scores[BLU] >= 2 or len(state.board.draft) == 0

    def getLegalActions(self, game_state, agent_id):
        actions = []
        agent_state = game_state.agents[agent_id]
        
        #First, give the agent the option to trade a dead card, if they haven't just done so.
        if not agent_state.trade:
            for card in agent_state.hand:
                if card[0]!='j':
                    free_spaces = 0
                    for r,c in COORDS[card]:
                        if game_state.board.chips[r][c]==EMPTY:
                            free_spaces+=1
                    if not free_spaces: #No option to place, so card is considered dead and can be traded.
                        for draft in game_state.board.draft:
                            actions.append({'play_card':card, 'draft_card':draft, 'type':'trade', 'coords':None})
                        
            if len(actions): #If trade actions available, return those, along with the option to forego the trade.
                actions.append({'play_card':None, 'draft_card':None, 'type':'trade', 'coords':None})
                return actions
                
        #If trade is prohibited, or no trades available, add action/s for each card in player's hand.
        #For each action, add copies corresponding to the various draft cards that could be selected at end of turn.
        for card in agent_state.hand:
            if card in ['jd','jc']: #two-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if game_state.board.chips[r][c]==EMPTY:
                            for draft in game_state.board.draft:
                                actions.append({'play_card':card, 'draft_card':draft, 'type':'place', 'coords':(r,c)})
                            
            elif card in ['jh','js']: #one-eyed jacks
                for r in range(10):
                    for c in range(10):
                        if game_state.board.chips[r][c]==agent_state.opp_colour:
                            for draft in game_state.board.draft:
                                actions.append({'play_card':card, 'draft_card':draft, 'type':'remove', 'coords':(r,c)})
            
            else: #regular cards
                for r,c in COORDS[card]:
                    if game_state.board.chips[r][c]==EMPTY:
                        for draft in game_state.board.draft:
                            actions.append({'play_card':card, 'draft_card':draft, 'type':'place', 'coords':(r,c)})
                    
        return actions

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
    # record a unique node id to distinguish duplicated states for visualisation
    nextNodeID = 0

    def __init__(self, simulator, parent, state):
        self.simulator = simulator
        self.parent = parent
        self.state = copy.deepcopy(state)
        self.id = Node.nextNodeID
        Node.nextNodeID += 1

        # the value and the total visits to this node
        self.visits = 0
        self.value = 0.0

    def getValue(self):
        """
        Return the value of this node
        """
        return self.value


class StateNode(Node):
    def __init__(self, parent, state, agent_id, reward=0):
        super().__init__(None, parent, state)

        # a dictionary from action to State Node
        self.children = {}

        # a dictionary from State Node to action
        self.reverse_children = {}

        # the reward received for this state
        self.reward = reward

        # agent id
        self.agent_id = agent_id

    def isFullyExpanded(self, simulator, state):
        """
        Return true if and only if all child actions have been expanded
        """
        validActions = simulator.getLegalActions(state, self.agent_id)
        # 去掉重复动作，否则当其中有重复动作(比如玩家手中有2张一样的手牌)，且此处认为"仍有动作待探索"时 (未探索的动作即重复的动作)
        # select函数会选择这个节点，但expand函数中会认为这个节点已经full expanded
        validActions = set([self.dict2tuple(a) for a in validActions])

        if len(validActions) == len(self.children):
            return True
        else:
            return False

    def select(self, simulator, state):
        if not self.isFullyExpanded(simulator, state):
            # print(f"{self.id} - 就选自己了")
            return self
        else:
            # print(f"{self.id} - 从孩子里面选")
            actions = list(self.children.keys())
            qValues = []
            for action_tuple in actions:
                # get the Q values from all outcome nodes
                qValues.append(self.children[action_tuple].getValue())

            # epsilon-greedy
            random_number = random.random()
            if random_number < 0.1:
                selected_action = random.choice(actions)
            else:
                max_q_value = -10000
                best_actions = []
                for action_tuple, q_value in zip(actions, qValues):
                    if q_value > max_q_value:
                        max_q_value = q_value
                        best_actions = [action_tuple]
                    elif q_value == max_q_value:
                        best_actions.append(action_tuple)
                if len(best_actions) == 0:  # 游戏结束时，best_actions是空的
                    return None
                selected_action = random.choice(best_actions)  # 从最好(Q(s,a)最大的, 可能有多个)的动作中随机选出一个
            child = self.children[selected_action]
            return child.select(simulator, child.state)

    def dict2tuple(self, dict):
        return tuple(dict.items())

    def tuple2dict(self, tuple):
        return dict(tuple)

    def expand(self, simulator, state, node=None):
        # randomly select an unexpanded action to expand
        legal_actions = simulator.getLegalActions(state, self.agent_id)

        # 要把action从dict类型转化为tuple类型，并去掉重复动作
        legal_actions_tuple = set([self.dict2tuple(a) for a in legal_actions])

        # 去掉已经探索过的动作
        valid_actions = legal_actions_tuple - self.children.keys()

        # 不再有可行的动作
        if len(valid_actions) == 0:
            print(f"玩家: {self.agent_id}，节点: {self.id}")
            print("expand期间发现没有可行的动作:")
            print(f"所有可行的动作集合[dict] ({len(legal_actions)})个")
            print(f"所有可行的动作集合[tuple] [set] ({len(legal_actions_tuple)})个:")
            # for i, action in enumerate(sorted([self.dict2tuple(a) for a in legal_actions_tuple])):
            #     print(i, action)
            print(f"到达孩子的动作集合 ({len(self.children)})个: {self.children.keys()}")
            print(f"valid actions: {valid_actions}")
            print(f"游戏状态如下:")
            MCTS.print_state(state, self.agent_id)


            print(f"父节点id: {self.parent.id}")
            # if node is not None:
            #     print(f"父节点通过以下动作达到当前状态:")
            #     print(self.parent.reverse_children[node])
            MCTS.print_state(state)

            return None
        action = random.choice(list(valid_actions))
        action = self.tuple2dict(action)  # 再将动作由tuple转回dict

        # choose an outcome (不再使用概率，直接用确定性转化)
        new_state, _ = simulator.generateSuccessor(copy.deepcopy(state), action, self.agent_id)
        # newChild = EnvironmentNode(self.simulator, self, self.state, action, self.agent_id)
        # print(f"expand: 选择动作{action}")
        new_child = StateNode(self, copy.deepcopy(new_state), self.agent_id)
        self.children[self.dict2tuple(action)] = new_child
        self.reverse_children[new_child] = self.dict2tuple(action)
        return new_child

    def backPropagate(self, reward):
        self.visits += 1
        self.value = self.value + ((self.reward + reward - self.value) / self.visits)
        # print(f"节点信息：{self.id} - value: {self.value} - visits: {self.visits}")

        if self.parent != None:
            self.parent.backPropagate(reward * 0.9)  # 0.9 is the discount factor

    def getQFunction(self):
        qValues = {}
        for action in self.children.keys():
            qValues[(self.state, action)] = round(self.children[action].getValue(), 3)
        return qValues

    def bestAction(self):
        """Choose the action with the largest q-value, random tie break."""
        q_values = self.getQFunction()
        max_q = -10000
        best_actions = []
        for (state, action), q_value in q_values.items():
            if q_value > max_q:
                max_q = q_value
                best_actions = [action]
            elif q_value == max_q:
                best_actions.append(action)
        return self.tuple2dict(random.choice(best_actions))


class MCTS:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.all_nodes = {}  # debug，检查用

    def mcts(self, simulator, state, agent_id, timeout=1, print_info=False):
        # exclude = set()  # 有些节点在expand的时候没有可行的动作，放到这里面

        episode_done = 0  # select, expand, simulate的轮数

        simulator.current_game_state = copy.deepcopy(state)
        simulator.current_agent_index = agent_id

        rootNode = StateNode(parent=None,
                             state=copy.deepcopy(state),
                             agent_id=agent_id)

        self.all_nodes[rootNode.id] = rootNode

        if print_info:
            print("刚开始mcts的时候")
            print(f"根节点: {rootNode.id}")
            self.print_state(rootNode.state)

        # startTime = int(time.time() * 1000)
        # currentTime = int(time.time() * 1000)

        # 使用simulation次数作为退出循环的条件

        if print_info:
            print("进入循环 Select, Expand, Simulate, BackProp")
        # while currentTime < startTime + timeout * 1000:
        while episode_done < 100:
            if print_info:
                print(f"玩家: {agent_id}")
                print("==> 当前局面：")
                self.print_state(simulator.current_game_state)

            if print_info:
                print(">>> select <<<")

            selectedNode = rootNode.select(simulator, copy.deepcopy(state))
            if selectedNode is None:  # 游戏结束时，无节点可选，直接退出循环
                break
            if print_info:
                print(f"selected_node.state:")
                self.print_state(selectedNode.state)
            # while selectedNode in exclude:
            #     selectedNode = rootNode.select(simulator, copy.deepcopy(state))
            if print_info:
                print(f"选中node: {selectedNode.id}")
            if selectedNode.id not in self.all_nodes:
                self.all_nodes[selectedNode.id] = selectedNode

            if not simulator.endState(selectedNode.state):
                if print_info:
                    print(">>> expand <<<")
                # 红绿方棋位置，所有玩家手牌，draft都相同的情况下，
                # 同样的node，在select和expand的过程中竟然会有不一样的legal actions
                # 集合
                child = selectedNode.expand(simulator,
                                            copy.deepcopy(selectedNode.state),
                                            selectedNode)
                # if child is None:
                #     break
                if child.id not in self.all_nodes:
                    self.all_nodes[child.id] = child
                if print_info:
                    print(f"得到node: {child.id}")
                    print(">>> simulate <<<")
                reward = self.simulate(simulator, child, self.agent_id)  # simulate的过程把children里面每个node的state都给更新了
                if print_info:
                    print(">> back prop <<<")
                child.backPropagate(reward)
            else:
                if print_info:
                    print("游戏它就这样结束了")

            currentTime = int(time.time() * 1000)

            if print_info:
                self.print_tree_node_info()

            episode_done += 1
            if print_info:
                print(f">>>>>>  episode done: {episode_done}    <<<<<<<")

        if print_info:
            print("mcts 结束了")
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

    @classmethod
    def print_state(self, state, agent_id=None):
        print("==== 红方占领 ===")
        print(state.board.plr_coords[RED])
        print("==== 蓝方占领 ===")
        print(state.board.plr_coords[BLU])
        print("==== hands ====")
        for i in range(4):
            print(i, state.agents[i].hand)
        print("==== draft ====")
        print(state.board.draft)
        print("==== board ====")
        print(BoardToString(state))
        if agent_id is not None:
            print(f"agent {agent_id}.trade: {state.agents[agent_id].trade}")

    def choose(self, simulator, state, agent_id):
        """random action."""
        legal_actions = simulator.getLegalActions(state, agent_id)
        # print(f"所有可行动作: {legal_actions}")
        return random.choice(legal_actions)

    def simulate(self, simulator, node, agent_id, print_info=False):
        """Simulate until a terminate state."""
        # TODO: 确保每次simulate之前，状态都是在"比较原始"的状态
        state = copy.deepcopy(node.state)  # 如果不复制的话，会导致外部传进来的node中的state被更新
        agent_index = agent_id  # 当前的agent id
        if print_info:
            self.print_state(state)
        simulator.current_game_state = state  # 强行赋值
        simulator.current_agent_index = agent_id
        cumulativeReward = 0.0
        depth = 0

        simulation_episode_done = 0

        while not simulator.endState(state) and not simulator.gameEnds():
            if print_info:
                print(f"--> 第{simulation_episode_done}轮simulation")
            # choose an action to execute
            if print_info:
                print('\n' + ">>" * 50 + '\n')
                print(f"agent {self.agent_id} 选个动作")
            action = self.choose(simulator, state, agent_index)
            if print_info:
                print(action)
                print(f"执行动作")
            # execute the action
            (state, reward, agent_index) = simulator.execute(state, action, agent_index)
            if print_info:
                print(f"目前agent index: {agent_index}")
            simulator.current_agent_index = agent_index
            if print_info:
                self.print_state(state)

            if simulator.endState(state):
                if print_info:
                    print("玩到一半发现游戏结束了")
                break

            # print(f"更新一下")
            # self.simulator.update(action)

            # discount the reward
            cumulativeReward += pow(0.9, depth) * reward  # 0.9 is the discount factor
            depth += 1

            # 其他3个玩家假设随机选择动作
            # 可以不随机，以提升simulation的效果
            # if print_info:
            #     print(f"其他三个玩家......")
            game_end = False
            while agent_index != agent_id:
                # print("当前状态")
                # self.print_state(state)
                if print_info:
                    print(f"玩家{agent_index}")
                actions = simulator.getLegalActions(state, agent_index)
                # 随机选择动作
                selected_action = random.choice(actions)
                if print_info:
                    print(f"selected action: {selected_action}")
                try:
                    state, _, agent_index = simulator.execute(state, selected_action, agent_index)
                    simulator.current_agent_index = agent_index
                    if print_info:
                        print("执行动作后，局面如下")
                        self.print_state(state)
                        print(f"agent_index: {agent_index}")
                    if simulator.endState(state):
                        if print_info:
                            print(f"来自玩家{agent_index}的消息：游戏结束了凹")
                        game_end = True
                        break
                except:
                    game_end = True

            if game_end:
                break

            simulation_episode_done += 1
        return cumulativeReward