from template import Agent
import random
import copy
from utils import *
from Sequence.sequence_model import *


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)

    def bfs_search(self, start_state, agent_id):
        # 设定一个目标, sample
        def goal_state(state):
            my_color = state.agents[agent_id].colour
            my_positions = state.board.plr_coords[my_color]
            my_start_positions = start_state.board.plr_coords[my_color]

            if len(my_positions) == 0:
                return True

            for pos in my_start_positions:
                for pos2 in my_positions:
                    if pos2 not in my_start_positions:
                        if abs(pos[0] - pos2[0]) + abs(pos[1] - pos[1]) <= 3:
                            return True

            return False

        # improvement:
        # 策略1： 只有自己在玩
        # 策略2： 考虑其他用户

        # 1. heuristic search
        # 2. define goal state
        # 3. Design heuristic function


        # 创建一个虚拟的游戏对象，游戏的开始状态是拼接后开始的游戏状态
        # begin_state = SequenceStateFromValue(start_state)
        simulator = SequenceGameRuleFromValue(copy.deepcopy(start_state), self.id, 4)
        begin_state = simulator.current_game_state

        queue = [(begin_state, [])]  # (state, action, sequence)
        while len(queue) > 0:
            cur_state, action_seq = queue.pop(0)
            valid_actions = simulator.getLegalActions(cur_state, agent_id)

            for action in valid_actions:
                # 不用generateSuccessor只生成新的状态，无法更新状态
                # next_state, _ = simulator.generateSuccessor(copy.deepcopy(cur_state), action, self.agent_id)
                # 策略2: 得到 agent_index
                # 用while loop加入其他用户的动作

                next_state, reward, cur_agent_index = simulator.execute(copy.deepcopy(cur_state), action, self.id)
                tmp_state = next_state
                while cur_agent_index != self.id:
                    new_actions = simulator.getLegalActions(tmp_state, cur_agent_index)
                    tmp_state, _ = simulator.execute(copy.deepcopy(cur_state), random.choice(new_actions), cur_agent_index)

                next_state = tmp_state

                if goal_state(next_state):
                    return action_seq + [action] #是目标状态
                queue.append(next_state, action_seq + [action]) # 不是goal state，加入到queue里面

    # Classical Planning 使用bfs search找到动作序列并返回动作系列中的第一个动作
    # 首先要定义目标函数，也就是判定一个目标状态
    # 整个搜索流程为，首先给定一个初始状态（新建一个模拟游戏的对象，每次寻找该用户可执行的所有动作后的新的状态，然后检测这个状态是否为goal state，
    # 如果不是，将放到队列的末尾。如果是，则直接返回动作序列
    def SelectAction(self, actions, game_state):
        my_actions = self.bfs_search(game_state, self.id)  # 动作序列
        if len(my_actions) == 0:
            return random.choice(actions)
        return my_actions[0]
