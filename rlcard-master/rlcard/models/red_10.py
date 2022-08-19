''' Dou Dizhu rule models
'''

import numpy as np
import os
import torch
from torch import nn
import rlcard
from rlcard.games.doudizhu.utils import CARD_TYPE, INDEX
from rlcard.models.model import Model
"""
class Red_10RuleAgentV1(object):
    ''' Dou Dizhu Rule agent version 1
    '''

    def __init__(self):
        self.use_raw = True

    def step(self, state):
        ''' Predict the action given raw state. A naive rule.
        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''
        state = state['raw_obs']
        trace = state['trace']
        # the rule of leading round
        if len(trace) == 0 or (len(trace) >= 4 and trace[-1][1] == 'pass' and trace[-2][1] == 'pass' and trace[-3][1] == 'pass'):
            comb = self.combine_cards(state['current_hand'])
            min_card = state['current_hand'][0]
            for _, actions in comb.items():
                for action in actions:
                    if min_card in action:
                        return action
        # the rule of following cards
        else:
            target = state['trace'][-1][1]
            target_player = state['trace'][-1][0]
            if target == 'pass':
                target = state['trace'][-2][1]
                target_player = state['trace'][-2][0]

            if target == 'pass':
                target = state['trace'][-3][1]
                target_player = state['trace'][-3][0]
            the_type = CARD_TYPE[0][target][0][0]
            chosen_action = ''
            rank = 1000
            for action in state['actions']:
                if action != 'pass' and the_type == CARD_TYPE[0][action][0][0]:
                    if int(CARD_TYPE[0][action][0][1]) < rank:
                        rank = int(CARD_TYPE[0][action][0][1])
                        chosen_action = action
            if chosen_action != '':
                return chosen_action
            landlord = state['landlord']
            if target_player not in landlord and state['self'] not in landlord:
                return 'pass'
            if target_player in landlord and state['self'] in landlord:
                return 'pass'
            return np.random.choice(state['actions'])

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []

    def combine_cards(self, hand):
        '''Get optimal combinations of cards in hand
        '''
        comb = {'rocket': [], 'bomb': [], 'trio': [], 'trio_chain': [],
                'solo_chain': [], 'pair_chain': [], 'pair': [], 'solo': []}
        # 1. pick rocket
        if hand[-2:] == 'BR':
            comb['rocket'].append('BR')
            hand = hand[:-2]
        # 2. pick bomb
        hand_cp = hand
        for index in range(len(hand_cp) - 3):
            if hand_cp[index] == hand_cp[index+3]:
                bomb = hand_cp[index: index+4]
                comb['bomb'].append(bomb)
                hand = hand.replace(bomb, '')
        # 3. pick trio and trio_chain
        hand_cp = hand
        for index in range(len(hand_cp) - 2):
            if hand_cp[index] == hand_cp[index+2]:
                trio = hand_cp[index: index+3]
                if len(comb['trio']) > 0 and INDEX[trio[-1]] < 12 and (INDEX[trio[-1]]-1) == INDEX[comb['trio'][-1][-1]]:
                    comb['trio'][-1] += trio
                else:
                    comb['trio'].append(trio)
                hand = hand.replace(trio, '')
        only_trio = []
        only_trio_chain = []
        for trio in comb['trio']:
            if len(trio) == 3:
                only_trio.append(trio)
            else:
                only_trio_chain.append(trio)
        comb['trio'] = only_trio
        comb['trio_chain'] = only_trio_chain
        # 4. pick solo chain
        hand_list = self.card_str2list(hand)
        chains, hand_list = self.pick_chain(hand_list, 1)
        comb['solo_chain'] = chains
        # 5. pick par_chain
        chains, hand_list = self.pick_chain(hand_list, 2)
        comb['pair_chain'] = chains
        hand = self.list2card_str(hand_list)
        # 6. pick pair and solo
        index = 0
        while index < len(hand) - 1:
            if hand[index] == hand[index+1]:
                comb['pair'].append(hand[index] + hand[index+1])
                index += 2
            else:
                comb['solo'].append(hand[index])
                index += 1
        if index == (len(hand) - 1):
            comb['solo'].append(hand[index])
        return comb

    @staticmethod
    def card_str2list(hand):
        hand_list = [0 for _ in range(15)]
        for card in hand:
            hand_list[INDEX[card]] += 1
        return hand_list

    @staticmethod
    def list2card_str(hand_list):
        card_str = ''
        cards = [card for card in INDEX]
        for index, count in enumerate(hand_list):
            card_str += cards[index] * count
        return card_str

    @staticmethod
    def pick_chain(hand_list, count):
        chains = []
        str_card = [card for card in INDEX]
        hand_list = [str(card) for card in hand_list]
        hand = ''.join(hand_list[:12])
        chain_list = hand.split('0')
        add = 0
        for index, chain in enumerate(chain_list):
            if len(chain) > 0:
                if len(chain) >= 5:
                    start = index + add
                    min_count = int(min(chain)) // count
                    if min_count != 0:
                        str_chain = ''
                        for num in range(len(chain)):
                            str_chain += str_card[start+num]
                            hand_list[start+num] = int(hand_list[start+num]) - int(min(chain))
                        for _ in range(min_count):
                            chains.append(str_chain)
                add += len(chain)
        hand_list = [int(card) for card in hand_list]
        return (chains, hand_list)


class Red_10RuleModelV1(Model):
    ''' Dou Dizhu Rule Model version 1
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('red_10')

        rule_agent = Red_10RuleAgentV1()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.rule_agents
"""
class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(216, 128, batch_first=True)
        self.dense1 = nn.Linear(594 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class Relation_score_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(216, 128, batch_first=True)
        self.dense1 = nn.Linear(491 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 3)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        x = torch.sigmoid(x)
        """
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)
        """
        return x

#get the rate of the dangerous(game is end)
class Dangerous_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(216, 128, batch_first=True)
        self.dense1 = nn.Linear(491 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        x = torch.sigmoid(x)
        """
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)
        """
        return x

model_dict = {}
model_dict['landlord'] = FarmerLstmModel
model_dict['landlord_up'] = FarmerLstmModel
model_dict['landlord_down'] = FarmerLstmModel
model_dict['landlord_front'] = FarmerLstmModel
model_dict['relation'] = Relation_score_Model
model_dict['dangerous'] = Dangerous_Model
pretrained_dir = os.path.join(rlcard.__path__[0], 'models/pretrained/red_10_pretrained')
pretrained_RD_dir = os.path.join(rlcard.__path__[0], 'models/pretrained/RD_pretrained')

#juge-policy
class Red_10RuleModelV1(Model):
    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('red_10')
        self.agent = {}
        for index, m in enumerate(['douzero_1000/model','douzero_1100/model','douzero_1010/model','douzero_1111/model']):
            players = {}
            for position in ['landlord', 'landlord_down', 'landlord_front', 'landlord_up']:
                pretrained_dir_=os.path.join(pretrained_dir,m)
                players[position] = (replayDeepAgent(position, pretrained_dir_, use_onnx=False, num_actions=env.num_actions))
            self.agent[str(index)] = players
        self.num_players = env.num_players

        self.relation = _load_model('relation',pretrained_RD_dir,use_onnx=False)
        self.dangerous = _load_model('dangerous',pretrained_RD_dir,use_onnx=False)
    def agents(self,rela_state=None,danger_state=None):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if rela_state and danger_state:
            rela_obs_x = torch.from_numpy(rela_state['obs_x']).to(device)
            rela_obs_z = torch.from_numpy(rela_state['obs_z']).to(device)
            relation_x = torch.unsqueeze(rela_obs_x, dim=0)
            relation_z = torch.unsqueeze(rela_obs_z, dim=0)
            relation_score = self.relation(relation_z,relation_x)

            danger_obs_x = torch.from_numpy(danger_state['obs_x']).to(device)
            danger_obs_z = torch.from_numpy(danger_state['obs_z']).to(device)
            danger_x = torch.unsqueeze(danger_obs_x, dim=0)
            danger_z = torch.unsqueeze(danger_obs_z, dim=0)
            danger_score = self.dangerous(danger_z,danger_x)

            relation_score = torch.squeeze(relation_score, dim=0).cpu().detach().numpy()
            dangerous_score = torch.squeeze(danger_score, dim=0).cpu().detach().numpy()
            agent, model_numble = self.decide_players(relation_score,dangerous_score,self.agent)
            r_sl = []
            for x in relation_score:
                r_sl.append(float(x))
            return r_sl, float(dangerous_score), model_numble, agent
        else:
            agent = []
            for p in ['landlord', 'landlord_down', 'landlord_front', 'landlord_up']:
                agent.append(self.agent['0'][p])
            return 0.0,0.0,1000,agent
    def decide_players(self,relation_score, danger_score, players):
        model_numbles = [1000, 1100, 1010, 1111]
        strnumber = ''
        countnumber = 0
        for x in relation_score:
            if x > danger_score:
                strnumber += '1'
                countnumber += 1
            else:
                strnumber += '0'
        if countnumber == 0:
            return players['0']['landlord'], model_numbles[0]
        elif countnumber == 1:
            if strnumber[0] == '1':
                return players['1']['landlord_down'], model_numbles[1]
            elif strnumber[1] == '1':
                return players['2']['landlord'], model_numbles[2]
            elif strnumber[2] == '1':
                return players['1']['landlord'], model_numbles[1]
            else:
                raise Exception
        elif countnumber == 2:
            if strnumber[0] == '0':
                return players['0']['landlord_down'], model_numbles[0]
            elif strnumber[1] == '0':
                return players['0']['landlord_front'], model_numbles[0]
            elif strnumber[2] == '0':
                return players['0']['landlord_up'], model_numbles[0]
            else:
                raise Exception
        elif countnumber == 3:
            if strnumber == '111':
                return players['3']['landlord'], model_numbles[3]
            else:
                raise Exception

#direct policy
class Red_10RuleModelV2(Model):
    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('red_10')
        self.agent = []
        players = {}
        for position in ['landlord', 'landlord_down', 'landlord_front', 'landlord_up']:
            pretrained_dir_=os.path.join(pretrained_dir,'douzero')
            players[position] = (replayDeepAgent(position, pretrained_dir_, use_onnx=False, num_actions=env.num_actions))
        self.agent = players
        self.num_players = env.num_players

    def agents(self,rela_state=None,danger_state=None):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        agent = []
        for p in ['landlord', 'landlord_down', 'landlord_front', 'landlord_up']:
            agent.append(self.agent[p])
        return 0.0,0.0,1000,agent

#remove the danger network
class Red_10RuleModelV3(Model):
    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('red_10')
        self.agent = {}
        for index, m in enumerate(['douzero_1000/model','douzero_1100/model','douzero_1010/model','douzero_1111/model']):
            players = {}
            for position in ['landlord', 'landlord_down', 'landlord_front', 'landlord_up']:
                pretrained_dir_=os.path.join(pretrained_dir,m)
                players[position] = (replayDeepAgent(position, pretrained_dir_, use_onnx=False, num_actions=env.num_actions))
            self.agent[str(index)] = players
        self.num_players = env.num_players

        self.relation = _load_model('relation',pretrained_RD_dir,use_onnx=False)
        self.dangerous = _load_model('dangerous',pretrained_RD_dir,use_onnx=False)
    def agents(self,rela_state=None,danger_state=None):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if rela_state and danger_state:
            rela_obs_x = torch.from_numpy(rela_state['obs_x']).to(device)
            rela_obs_z = torch.from_numpy(rela_state['obs_z']).to(device)
            relation_x = torch.unsqueeze(rela_obs_x, dim=0)
            relation_z = torch.unsqueeze(rela_obs_z, dim=0)
            relation_score = self.relation(relation_z,relation_x)

            relation_score = torch.squeeze(relation_score, dim=0).cpu().detach().numpy()

            agent, model_numble = self.decide_players_nodanger(relation_score,self.agent)
            r_sl = []
            for x in relation_score:
                r_sl.append(float(x))
            return r_sl, float(0.5), model_numble, agent
        else:
            agent = []
            for p in ['landlord', 'landlord_down', 'landlord_front', 'landlord_up']:
                agent.append(self.agent['0'][p])
            return 0.0,0.5,1000, agent
    def decide_players(self,relation_score, danger_score, players):
        model_numbles = [1000, 1100, 1010, 1111]
        strnumber = ''
        countnumber = 0
        for x in relation_score:
            if x > danger_score:
                strnumber += '1'
                countnumber += 1
            else:
                strnumber += '0'
        if countnumber == 0:
            return players['0']['landlord'], model_numbles[0]
        elif countnumber == 1:
            if strnumber[0] == '1':
                return players['1']['landlord_down'], model_numbles[1]
            elif strnumber[1] == '1':
                return players['2']['landlord'], model_numbles[2]
            elif strnumber[2] == '1':
                return players['1']['landlord'], model_numbles[1]
            else:
                raise Exception
        elif countnumber == 2:
            if strnumber[0] == '0':
                return players['0']['landlord_down'], model_numbles[0]
            elif strnumber[1] == '0':
                return players['0']['landlord_front'], model_numbles[0]
            elif strnumber[2] == '0':
                return players['0']['landlord_up'], model_numbles[0]
            else:
                raise Exception
        elif countnumber == 3:
            if strnumber == '111':
                return players['3']['landlord'], model_numbles[3]
            else:
                raise Exception

    def decide_players_nodanger(self,relation_score, players):
        model_numbles = [1000, 1100, 1010, 1111]
        danger_score = 0.5
        strnumber = ''
        countnumber = 0
        for x in relation_score:
            if x > danger_score:
                strnumber += '1'
                countnumber += 1
            else:
                strnumber += '0'
        if countnumber == 0:
            return players['0']['landlord'], model_numbles[0]
        elif countnumber == 1:
            if strnumber[0] == '1':
                return players['1']['landlord_down'], model_numbles[1]
            elif strnumber[1] == '1':
                return players['2']['landlord'], model_numbles[2]
            elif strnumber[2] == '1':
                return players['1']['landlord'], model_numbles[1]
            else:
                raise Exception
        elif countnumber == 2:
            if strnumber[0] == '0':
                return players['0']['landlord_down'], model_numbles[0]
            elif strnumber[1] == '0':
                return players['0']['landlord_front'], model_numbles[0]
            elif strnumber[2] == '0':
                return players['0']['landlord_up'], model_numbles[0]
            else:
                raise Exception
        elif countnumber == 3:
            if strnumber == '111':
                return players['3']['landlord'], model_numbles[3]
            else:
                raise Exception

def _load_model(position, model_dir, use_onnx):
    import os

    if not use_onnx or not os.path.isfile(os.path.join(model_dir, position+'.onnx')) :
        model = model_dict[position]()
        model_state_dict = model.state_dict()
        model_path = os.path.join(model_dir, position+'_weights.ckpt')
        if torch.cuda.is_available():
            pretrained = torch.load(model_path, map_location='cuda:0')
        else:
            pretrained = torch.load(model_path, map_location='cpu')
        pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        if use_onnx:
            z = torch.randn(1, 5, 216, requires_grad=True)
            x = torch.randn(1, 594, requires_grad=True)
            torch.onnx.export(model,
                              (z,x),
                              os.path.join(model_dir, position+'.onnx'),
                              export_params=True,
                              opset_version=10,
                              do_constant_folding=True,
                              input_names = ['z', 'x'],
                              output_names = ['y'],
                              dynamic_axes={'z' : {0 : 'batch_size'},
                                            'x' : {0 : 'batch_size'},
                                            'y' : {0 : 'batch_size'}})

    if use_onnx:
        import onnxruntime
        model = onnxruntime.InferenceSession(os.path.join(model_dir, position+'.onnx'))
    return model

class replayDeepAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, position, pretrained_dir, use_onnx=False,num_actions=0):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw =  True
        self.model = _load_model(position, pretrained_dir, use_onnx)
        self.num_actions = num_actions
        self.use_onnx = use_onnx

    @staticmethod
    def step(self,state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (str): The action predicted (randomly chosen) by the pretrained  agent
        '''
        z_batch = state['z_batch']
        x_batch = state['x_batch']
        if self.use_onnx:
            ort_inputs = {'z': z_batch, 'x': x_batch}
            y_pred = self.model.run(None, ort_inputs)[0]
        elif torch.cuda.is_available():
            y_pred = self.model.forward(torch.from_numpy(z_batch).float().cuda(),
                                        torch.from_numpy(x_batch).float().cuda(),return_value=True)['values']
            y_pred = y_pred.cpu().detach().numpy()
        else:
            y_pred = self.model.forward(torch.from_numpy(z_batch).float(),
                                        torch.from_numpy(x_batch).float(),return_value=True)['values']
            y_pred = y_pred.detach().numpy()

        y_pred = y_pred.flatten()

        first_action_index = np.argpartition(y_pred, -1)[-1:][0]
        size = min(3, len(y_pred))
        best_action_index = np.argpartition(y_pred, -size)[-size:]
        best_action_confidence = y_pred[best_action_index]
        best_action = [state['raw_obs']['actions'][index] for index in best_action_index]
        choice_action = state['raw_obs']['actions'][first_action_index]
        return  choice_action, best_action,best_action_confidence

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        choice_action,best_action, best_action_confidence = self.step(self,state = state)
        info = {}
        info['probs'] = {}
        for action,action_c in zip(best_action,best_action_confidence):
            info['probs'][action] = action_c
        return choice_action, info