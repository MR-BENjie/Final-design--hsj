"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch 

def _format_observation(obs, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           'raw_legal_actions': obs ['raw_legal_actions']
           }
    return position, obs, x_no_action, z

def _format_R_D_obs(relation_state, danger_state ,device):
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)

    rela_obs_x = torch.from_numpy(relation_state['obs_x']).to(device)
    rela_obs_z = torch.from_numpy(relation_state['obs_z']).to(device)
    rela_target = torch.from_numpy(relation_state['teammate']).to(device)
    rela_c_color = relation_state['raw_obs']["current_color"]

    danger_obs_x = torch.from_numpy(danger_state['obs_x']).to(device)
    danger_obs_z = torch.from_numpy(danger_state['obs_z']).to(device)
    danger_target = danger_state['round_num']
    danger_c_color = danger_state['raw_obs']["current_color"]

    rela_obs = {'x': rela_obs_x,
           'z': rela_obs_z,
           'target':rela_target,
           'legal_actions':relation_state['raw_legal_actions'],
           'red_10_color':rela_c_color,
           }
    danger_obs = {'x': danger_obs_x,
                'z': danger_obs_z,
                'target': danger_target,
                'legal_actions': danger_state['raw_legal_actions'],
                'red_10_color':danger_c_color
                }

    return rela_obs, danger_obs


class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        initial_position, initial_obs, x_no_action, z = _format_observation(self.env.reset(), self.device)
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            )
        
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)

        self.episode_return += reward
        episode_return = self.episode_return 

        if done:
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)

        position, obs, x_no_action, z = _format_observation(obs, self.device)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        
        return position, obs, dict(
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            )

    def close(self):
        self.env.close()


class Environment_R_D:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        relation_state, danger_state, player_id , player_state = self.env.reset()

        initial_position = player_id
        obs_rela, obs_danger = _format_R_D_obs(relation_state,danger_state,self.device)
        _, obs_players, _, _ = _format_observation(player_state, self.device)

        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return initial_position, dict(
            done=initial_done,
            obs_relation = obs_rela,
            obs_dangerous = obs_danger,
            obs_player = obs_players
        )

    def step(self, action,color_input=None):
        relation_state, danger_state,player_state, done = self.env.step(action, color = color_input)

        if done:
            relation_state, danger_state, player_id, player_state  = self.env.reset()

        obs_rela, obs_danger = _format_R_D_obs(relation_state,danger_state,self.device)
        _, obs_players, _, _ = _format_observation(player_state, self.device)

        done = torch.tensor(done).view(1, 1)
        return  dict(
            done=done,
            obs_relation = obs_rela,
            obs_dangerous = obs_danger,
            obs_player = obs_players
        )

    def close(self):
        self.env.close()