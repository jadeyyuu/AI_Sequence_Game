from Sequence.sequence_utils import *
from Sequence.sequence_model import *
# 从半截开始的游戏对象
# 功能：从外部传一个游戏进来，然后开始一个游戏

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
        cur_user_hand = None
        for agent_state in self.agents:
            if hasattr(agent_state, 'hand'):  #agent_state.hand
                cur_user_hand = agent_state.hand

        # 全部的牌 按照默认的游戏流程创建一套完整的牌
        all_cards = [(r + s) for r in ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a'] for s in
                     ['d', 'c', 'h', 's']] * 2  # Sequence uses 2 decks.

        # 去掉当前玩家手中的牌
        for c in cur_user_hand:
            if c in all_cards:
                all_cards.remove(c)

        # 去掉整个游戏过程中已经丢弃的牌
        for c in self.deck.discards:
            if c in all_cards:
                all_cards.remove(c)

        # 去掉公共牌池的牌
        for c in self.board.draft:
            if c in all_cards:
                all_cards.remove(c)

        # 剩下的牌池(与真实的基本上完全一致,需去除其他三个用户的手牌)
        self.deck.cards = all_cards

        # 从剩下的牌中随机给其他3位玩家每人发6张牌
        for i, agent_state in enumerate(self.agent):
            if not hasattr(agent_state, 'hand'): # 如果没有hand属性，证明是"其他玩家"
                # 发牌过程会自动从当前牌池中删除掉已经发出去的牌， 所以不需要手动从deck.cards中删除已经发出去的牌
                # 具体可阅读Deck类中deal方法
                # 注：pop()方法会得到列表中的一个元素，并将其从列表中删除
                self.agent(i).hand = self.deck.deal(6)

class SequenceGameRuleFromValue(SequenceGameRule):
    """Game rule that can be created from a specified game state and agent index."""
    def __init__(self, state, agent_index, num_of_agent=4):
        super().__init__(num_of_agent)
        self.perfect_information = True
        self.current_agent_index = agent_index
        self.num_of_agent = num_of_agent
        self.current_game_state = SequenceStateFromValue(state)  # 使用我们手动创建的"虚拟游戏状态"
        self.action_counter = 0

    def generateSuccessor(self, state, action, agent_id):
        state.board.new_seq = False
        # print(f"agent id {agent_id}")
        plr_state = state.agents[agent_id]
        plr_state.last_action = action  # Record last action such that other agents can make use of this information.
        reward = 0

        # Update agent state. Take the card in play from the agent, discard, draw the selected draft, deal a new draft.
        # If agent was allowed to trade but chose not to, there is no card played, and hand remains the same.
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
                reward += seq['num_seq']
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

        plr_state.trade = False  # Reset trade flag if agent has completed a full turn.
        plr_state.agent_trace.action_reward.append((action, reward))  # Log this turn's action and any resultant score.
        plr_state.score += reward
        return state, reward

    #  可以考虑其他用户的操作，模拟游戏的流程，每一次让下一个用户去玩。如果是死牌，还是当前这个用户玩
    #  和update的代码很像，但是不会真正修改状态
    def execute(self, state, action, agent_index):
        temp_state = state
        new_state, reward = self.generateSuccessor(temp_state, action, agent_index)
        # if current action is to trade, agent in play continues their turn.
        current_agent_index = self.getNextAgentIndex() if action['type'] != 'trade' else self.current_agent_index
        self.action_counter += 1 #不重要

        # 返回：新的状态，得到的reward，新的状态下该出动作的agent index
        return new_state, reward, current_agent_index

    def endState(self, state):
        """检测是否是游戏终止状态"""
        scores = {RED: 0, BLU: 0}
        for plr_state in state.agents:
            scores[plr_state.colour] += plr_state.completed_seqs
        return scores[RED] >= 2 or scores[BLU] >= 2 or len(state.board.draft) == 0