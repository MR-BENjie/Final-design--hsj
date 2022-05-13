import rlcard
from rlcard.agents import RandomAgent
from rlcard.models.model import Model
import torch
from torch import nn
import numpy as np
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
# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['landlord'] = FarmerLstmModel
model_dict['landlord_up'] = FarmerLstmModel
model_dict['landlord_down'] = FarmerLstmModel
model_dict['landlord_front'] = FarmerLstmModel
pretrained_dir = 'pretrained/red_10_pretrained'

class Red_10RandomModelSpec(object):
    def __init__(self):
        self.model_id = 'red_10-random'
        self._entry_point = Red_10RandomModel

    def load(self):
        model = self._entry_point()
        return model

class Red_10RandomModel(Model):
    ''' A random model
    '''

    def __init__(self):
        ''' Load random model
        '''
        env = rlcard.make('red_10')
        self.agent = RandomAgent(num_actions=env.num_actions)
        self.num_players = env.num_players

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return [self.agent for _ in range(self.num_players)]

    @property
    def use_raw(self):
        ''' Indicate whether use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return False

class Red_10TrainedModelSpec(object):
    def __init__(self):
        self.model_id = 'red_10-trained'
        self._entry_point = Red_10TrainedModel

    def load(self):
        model = self._entry_point()
        return model

class Red_10TrainedModel(Model):
    ''' A random model
    '''

    def __init__(self):
        ''' Load random model
        '''
        env = rlcard.make('red_10')
        self.agents = []

        for position in ['landlord', 'landlord_down', 'landlord_front', 'landlord_up']:
            self.agents.append(replayDeepAgent(position, pretrained_dir, use_onnx=False,num_actions=env.num_actions))

        self.num_players = env.num_players

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.agents

    @property
    def use_raw(self):
        ''' Indicate whether use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return False

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
        self.use_raw = use_onnx

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
                                        torch.from_numpy(x_batch).float().cuda(),return_value=True)
            y_pred = y_pred.cpu().detach().numpy()
        else:
            y_pred = self.model.forward(torch.from_numpy(z_batch).float(),
                                        torch.from_numpy(x_batch).float(),return_value=True)
            y_pred = y_pred.detach().numpy()

        y_pred = y_pred.flatten()

        #best_action_index = np.argmax(y_pred, axis=0)[0]
        size = min(3, len(y_pred))
        best_action_index = np.argpartition(y_pred, -size)[-size:]
        best_action_confidence = y_pred[best_action_index]
        best_action = [state['legal_actions'][index] for index in best_action_index]
        choice_action = state['legal_actions'][np.argmax(y_pred, axis=0)[0]]
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
        choice_action,best_action, best_action_confidence = self.step(state)
        info = {}
        for action,action_c in zip(best_action,best_action_confidence):
            info[action] = action_c
        return choice_action, info