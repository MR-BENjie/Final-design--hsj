from collections import Counter, OrderedDict
import numpy as np

from rlcard.envs import Env


class Red_10Env(Env):
    ''' Doudizhu Environment
    '''

    def __init__(self, config):
        from rlcard.games.red_10.utils import ACTION_2_ID, ID_2_ACTION
        from rlcard.games.red_10.utils import cards2str, cards2str_with_suit
        from rlcard.games.red_10 import Game
        self._cards2str = cards2str
        self._cards2str_with_suit = cards2str_with_suit
        self._ACTION_2_ID = ACTION_2_ID
        self._ID_2_ACTION = ID_2_ACTION
        
        self.name = 'red_10'
        self.game = Game()
        super().__init__(config)
        self.state_shape = [[790], [901], [901]]
        self.action_shape = [[54] for _ in range(self.num_players)]

    #构造red-10 model的 state 矩阵（需要按照实际需求重新设计编写）
    def _extract_red_10_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state
        '''

        legal_actions = state['actions']
        num_legal_actions = len(legal_actions)



        current_hand = _cards2array(state['current_hand'])
        current_hand_batch = np.repeat(
            current_hand[np.newaxis, :],
            num_legal_actions, axis=0)

        my_action_batch = np.zeros(current_hand_batch.shape)
        for j, action in enumerate(legal_actions):
            my_action_batch[j, :] = _cards2array(action)


        others_hand = _cards2array(state['others_hand'])
        others_hand_batch = np.repeat(
            others_hand[np.newaxis, :],
            num_legal_actions, axis=0)

        last_action = ''
        if len(state['trace']) != 0:
            if state['trace'][-1][1] == 'pass':
                last_action = state['trace'][-2][1]
            else:
                last_action = state['trace'][-1][1]
        last_action = _cards2array(last_action)
        last_action_batch = np.repeat(
            last_action[np.newaxis, :],
            num_legal_actions, axis=0)

        last_20_actions = _action_seq2array(_process_action_seq(state['trace'],length=20))

        teammate_u_id = (state['self']-1)%4
        teammate_u_played_cards = _cards2array(state['played_cards'][teammate_u_id])
        teammate_u_played_cards_batch = np.repeat(
            teammate_u_played_cards[np.newaxis, :],
            num_legal_actions, axis=0)

        last_teammate_u_action = 'pass'
        for i, action, _ in reversed(state['trace']):
            if i == teammate_u_id:
                last_teammate_u_action = action
        last_teammate_u_action = _cards2array(last_teammate_u_action)
        last_teammate_u_action_batch = np.repeat(
            last_teammate_u_action[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_u_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_u_id], 13)
        teammate_u_num_cards_left_batch = np.repeat(
            teammate_u_num_cards_left[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_f_id = (state['self'] - 2) % 4
        teammate_f_played_cards = _cards2array(state['played_cards'][teammate_f_id])
        teammate_f_played_cards_batch = np.repeat(
            teammate_f_played_cards[np.newaxis, :],
            num_legal_actions, axis=0)

        last_teammate_f_action = 'pass'
        for i, action, _ in reversed(state['trace']):
            if i == teammate_f_id:
                last_teammate_f_action = action
        last_teammate_f_action = _cards2array(last_teammate_f_action)
        last_teammate_f_action_batch = np.repeat(
            last_teammate_f_action[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_f_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_f_id], 13)
        teammate_f_num_cards_left_batch = np.repeat(
            teammate_f_num_cards_left[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_d_id = (state['self'] - 3) % 4
        teammate_d_played_cards = _cards2array(state['played_cards'][teammate_d_id])
        teammate_d_played_cards_batch = np.repeat(
            teammate_d_played_cards[np.newaxis, :],
            num_legal_actions, axis=0)

        last_teammate_d_action = 'pass'
        for i, action, _ in reversed(state['trace']):
            if i == teammate_d_id:
                last_teammate_d_action = action
        last_teammate_d_action = _cards2array(last_teammate_d_action)
        last_teammate_d_action_batch = np.repeat(
            last_teammate_d_action[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_d_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_d_id], 13)
        teammate_d_num_cards_left_batch = np.repeat(
            teammate_d_num_cards_left[np.newaxis, :],
            num_legal_actions, axis=0)

        bomb_num = _get_one_hot_array(1, 15)
        bomb_num_batch = np.repeat(
            bomb_num[np.newaxis, :],
            num_legal_actions, axis=0)

        obs_x = np.hstack((current_hand_batch,
                        others_hand_batch,
                        teammate_u_played_cards_batch,
                        teammate_f_played_cards_batch,
                        teammate_d_played_cards_batch,
                        last_action_batch,
                        last_teammate_u_action_batch,
                        last_teammate_f_action_batch,
                        last_teammate_d_action_batch,
                        teammate_u_num_cards_left_batch,
                        teammate_f_num_cards_left_batch,
                        teammate_d_num_cards_left_batch,
                        bomb_num_batch,
                        my_action_batch))

        z = last_20_actions
        obs_z = np.repeat(
            z[np.newaxis, :, :],
            num_legal_actions, axis=0)

        extracted_state = OrderedDict({'x_batch': obs_x.astype(np.float32),'z_batch':obs_z.astype(np.float32), 'legal_actions': self._get_legal_actions()})
        extracted_state['raw_obs'] = state
        extracted_state['position'] = "landlord_123"
        extracted_state['raw_legal_actions'] = [a for a in state['actions']]
        extracted_state['action_record'] = self.action_recorder
        extracted_state['x_no_action'] = obs_x.astype(np.float32)
        extracted_state['z'] = obs_z.astype(np.float32)
        """
        obs_x和obs_z的使用，其中model是red-10的出牌模型,输入obs获得要执行的action在legal_action的idx.
        with torch.no_grad():
            agent_output = model.forward(position, obs_z, obs_x, flags=flags(None))
        _action_idx = int(agent_output['action'].cpu().detach().numpy())
        action = obs['legal_actions'][_action_idx]
        """
        return extracted_state

    # 构造relation_decition.py 中 Relation_score_Model 的 state 矩阵（需要按照实际需求重新设计编写）
    def _extract_relation_score_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state
        '''

        current_hand = _cards2array(state['current_hand'])


        played_red_10_color_list = state["played_red_10_color"]

        others_hand = _cards2array(state['others_hand'])

        last_20_actions = _action_seq2array(_process_action_seq(state['trace'],length=20))

        teammate_u_id = (state['self']-1)%4
        teammate_u_played_cards = _cards2array(state['played_cards'][teammate_u_id])

        last_teammate_u_action = 'pass'
        for i, action, _ in reversed(state['trace']):
            if i == teammate_u_id:
                last_teammate_u_action = action
        last_teammate_u_action = _cards2array(last_teammate_u_action)

        teammate_u_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_u_id], 13)

        teammate_u_played_color = color2_one_hot_array(played_red_10_color_list[teammate_u_id])

        teammate_f_id = (state['self'] - 2) % 4
        teammate_f_played_cards = _cards2array(state['played_cards'][teammate_f_id])

        last_teammate_f_action = 'pass'
        for i, action, _ in reversed(state['trace']):
            if i == teammate_f_id:
                last_teammate_f_action = action
        last_teammate_f_action = _cards2array(last_teammate_f_action)

        teammate_f_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_f_id], 13)

        teammate_f_played_color = color2_one_hot_array(played_red_10_color_list[teammate_f_id])


        teammate_d_id = (state['self'] - 3) % 4
        teammate_d_played_cards = _cards2array(state['played_cards'][teammate_d_id])


        last_teammate_d_action = 'pass'
        for i, action, _ in reversed(state['trace']):
            if i == teammate_d_id:
                last_teammate_d_action = action
        last_teammate_d_action = _cards2array(last_teammate_d_action)

        teammate_d_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_d_id], 13)

        teammate_d_played_color = color2_one_hot_array(played_red_10_color_list[teammate_d_id])

        current_hand_playerd_color = color2_one_hot_array(played_red_10_color_list[state['self']])
        current_hand_playerd_color+= color2_one_hot_array(state["current_color"])


        other_hand_played_color = np.ones(4, dtype=np.int8)-current_hand_playerd_color-teammate_d_played_color-teammate_u_played_color-\
                                  teammate_f_played_color


        obs_x = np.concatenate((current_hand,
                        others_hand,
                        teammate_u_played_cards,
                        teammate_f_played_cards,
                        teammate_d_played_cards,
                        last_teammate_u_action,
                        last_teammate_f_action,
                        last_teammate_d_action,
                        teammate_u_num_cards_left,
                        teammate_f_num_cards_left,
                        teammate_d_num_cards_left,
                        teammate_u_played_color,
                        teammate_f_played_color,
                        teammate_d_played_color,
                        current_hand_playerd_color,
                        other_hand_played_color))

        z = last_20_actions
        obs_z = z

        teammate_player_vector = np.zeros(3)
        if state['self'] in state['landlord']:
            for l in state['landlord']:
                if l == teammate_u_id :
                    teammate_player_vector[0] = 1
                elif l == teammate_f_id:
                    teammate_player_vector[1] = 1
                elif l == teammate_d_id:
                    teammate_player_vector[2] = 1
        else :
            if teammate_u_id not in state['landlord']:
                teammate_player_vector[0] = 1
            elif teammate_f_id not in state['landlord']:
                teammate_player_vector[1] = 1
            elif teammate_d_id not in state['landlord']:
                teammate_player_vector[2] = 1

        extracted_state = OrderedDict({'obs_x': obs_x.astype(np.float32),'obs_z':obs_z.astype(np.float32), 'legal_actions': self._get_legal_actions()})
        extracted_state['teammate'] = teammate_player_vector
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['actions']]
        extracted_state['action_record'] = self.action_recorder

        """
        obs_x和obs_z的使用，其中model是Relation_score_Model的置信度计算模型.
        with torch.no_grad():
            agent_output = model.forward(position, obs_z, obs_x, flags=flags(None))
        """
        return extracted_state

    # 构造relation_decition.py 中 Dangerous_Model 的 state 矩阵（需要按照实际需求重新设计编写）
    def _extract_dangerous_state(self, state):
        ''' Encode state

        Args:
            state (dict): dict of original state
        '''
        current_hand = _cards2array(state['current_hand'])


        played_red_10_color_list = state["played_red_10_color"]

        others_hand = _cards2array(state['others_hand'])


        last_20_actions = _action_seq2array(_process_action_seq(state['trace'],length=20))

        teammate_u_id = (state['self']-1)%4
        teammate_u_played_cards = _cards2array(state['played_cards'][teammate_u_id])

        last_teammate_u_action = 'pass'
        for i, action,_ in reversed(state['trace']):
            if i == teammate_u_id:
                last_teammate_u_action = action
        last_teammate_u_action = _cards2array(last_teammate_u_action)


        teammate_u_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_u_id], 13)


        teammate_u_played_color = color2_one_hot_array(played_red_10_color_list[teammate_u_id])


        teammate_f_id = (state['self'] - 2) % 4
        teammate_f_played_cards = _cards2array(state['played_cards'][teammate_f_id])


        last_teammate_f_action = 'pass'
        for i, action,_ in reversed(state['trace']):
            if i == teammate_f_id:
                last_teammate_f_action = action
        last_teammate_f_action = _cards2array(last_teammate_f_action)


        teammate_f_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_f_id], 13)

        teammate_f_played_color = color2_one_hot_array(played_red_10_color_list[teammate_f_id])


        teammate_d_id = (state['self'] - 3) % 4
        teammate_d_played_cards = _cards2array(state['played_cards'][teammate_d_id])

        last_teammate_d_action = 'pass'
        for i, action,_  in reversed(state['trace']):
            if i == teammate_d_id:
                last_teammate_d_action = action
        last_teammate_d_action = _cards2array(last_teammate_d_action)

        teammate_d_num_cards_left = _get_one_hot_array(state['num_cards_left'][teammate_d_id], 13)

        teammate_d_played_color = color2_one_hot_array(played_red_10_color_list[teammate_d_id])


        current_hand_playerd_color = color2_one_hot_array(played_red_10_color_list[state['self']])
        current_hand_playerd_color+= color2_one_hot_array(state["current_color"])


        other_hand_played_color = np.ones(4,dtype=np.int8)-current_hand_playerd_color-teammate_d_played_color-teammate_u_played_color-\
                                  teammate_f_played_color

        obs_x = np.hstack((current_hand,
                        others_hand,
                        teammate_u_played_cards,
                        teammate_f_played_cards,
                        teammate_d_played_cards,
                        last_teammate_u_action,
                        last_teammate_f_action,
                        last_teammate_d_action,
                        teammate_u_num_cards_left,
                        teammate_f_num_cards_left,
                        teammate_d_num_cards_left,
                        teammate_u_played_color,
                        teammate_f_played_color,
                        teammate_d_played_color,
                        current_hand_playerd_color,
                        other_hand_played_color))

        z = last_20_actions
        obs_z = z

        extracted_state = OrderedDict({'obs_x': obs_x.astype(np.float32),'obs_z':obs_z.astype(np.float32), 'legal_actions': self._get_legal_actions()})
        extracted_state['round_num'] = len(state['trace'])+1
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['actions']]
        extracted_state['action_record'] = self.action_recorder

        """
        obs_x和obs_z的使用，其中model是Dangerous_Model的置信度计算模型.
        with torch.no_grad():
            agent_output = model.forward(position, obs_z, obs_x, flags=flags(None))
        """
        return extracted_state

    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        return self.game.judger.judge_payoffs(self.game.round.landlord_id, self.game.winner_id)

    def _decode_action(self, action_id):
        ''' Action id -> the action in the game. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action (string): the action that will be passed to the game engine.
        '''
        return self._ID_2_ACTION[action_id]

    def _get_legal_actions(self):
        ''' Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        legal_actions = self.game.state['actions']
        legal_actions = {self._ACTION_2_ID[action]: _cards2array(action) for action in legal_actions}
        return legal_actions

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['hand_cards_with_suit'] = [self._cards2str_with_suit(player.current_hand) for player in self.game.players]
        state['hand_cards'] = [self._cards2str(player.current_hand) for player in self.game.players]
        state['trace'] = self.game.state['trace']
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.state['actions']
        return state

    def get_action_feature(self, action):
        ''' For some environments such as DouDizhu, we can have action features

        Returns:
            (numpy.array): The action features
        '''
        return _cards2array(self._decode_action(action))

Card2Column = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7,
               'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

def _cards2array(cards):
    if cards == 'pass':
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(cards)
    for card, num_times in counter.items():
        if card == 'B':
            jokers[0] = 1
        elif card == 'R':
            jokers[1] = 1
        else:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
    return np.concatenate((matrix.flatten('F'), jokers))

def _get_one_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards, dtype=np.int8)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _action_seq2array(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 216)
    return action_seq_array

def _process_action_seq(sequence, length=9):
    sequence = [action[1] for action in sequence[-length:]]
    if len(sequence) < length:
        empty_sequence = ['' for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def color2_one_hot_array(color):
        colors = np.zeros(4,dtype=np.int8)
        if color=="" or color == []:
            return colors
        else:
            for co in color:
                if co == "S":
                    colors[0] = 1
                elif co == "H":
                    colors[1] = 1
                elif co == "D":
                    colors[2] = 1
                elif co == "C":
                    colors[3] = 1
            return colors