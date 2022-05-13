import os 
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time

import torch 
from torch import multiprocessing as mp

from .env_utils import Environment
from .env_utils import Environment_R_D
from douzero.env import Env
from douzero.env.env import _cards2array
from douzero.env.env import _cards2array_R_D
from douzero.env.rlcard_red10.rlcard.envs.red_10 import Red_10Env

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}


Card2Column_R_D = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, 'T': 7,
               'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12}


shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_env(flags):
    return Env(flags.objective)

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['landlord', 'landlord_up', 'landlord_down','landlord_front']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers

def create_R_D_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['relation','dangerous']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers

def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = ['landlord', 'landlord_up', 'landlord_down','landlord_front']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 540
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 54), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 216), dtype=torch.int8),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

def create_R_D_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = ['relation', 'dangerous']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 491
            if position == "relation":
                specs = dict(
                    done=dict(size=(T,), dtype=torch.bool),
                    target=dict(size=(T,3), dtype=torch.float32),
                    obs_x=dict(size=(T, x_dim), dtype=torch.int8),
                    obs_action=dict(size=(T, 54), dtype=torch.int8),
                    obs_z=dict(size=(T, 5, 216), dtype=torch.int8),
                )
            else:
                specs = dict(
                    done=dict(size=(T,), dtype=torch.bool),
                    target=dict(size=(T, ), dtype=torch.float32),
                    obs_x=dict(size=(T, x_dim), dtype=torch.int8),
                    obs_action=dict(size=(T, 54), dtype=torch.int8),
                    obs_z=dict(size=(T, 5, 216), dtype=torch.int8),
                )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

def decide_model(env,players):
    game_players = env.env.game.players
    str_p = ''
    count_p = 0
    for g_p in game_players:
        if g_p.role == 'landlord':
            str_p += '1'
            count_p += 1
        else:
            str_p += '0'

    p_model = {}
    positions = ['landlord', 'landlord_down', 'landlord_front', 'landlord_up']
    if count_p == 1:
        for index, p in enumerate(game_players):
            if p.role == 'landlord':
                for i in range(4):
                    p_model[(index+i)%4]=players[0][positions[i]]
                break
    elif count_p == 2:
        if int(str_p[0])+int(str_p[2])==1:
            for index, p in enumerate(game_players):
                if p.role == 'landlord':
                    if str_p[(index+1)%4] == '0':
                        for i in range(4):
                            p_model[(index + i) % 4] = players[1][positions[(i+1)%4]]
                    else:
                        for i in range(4):
                            p_model[(index + i) % 4] = players[1][positions[i%4]]

        else:
            for i in range(4):
                p_model[i] = players[2][positions[i]]
    return p_model

def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['landlord', 'landlord_up', 'landlord_down', 'landlord_front']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()

        while True:
            while True:
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])
                with torch.no_grad():
                    agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                size[position] += 1
                position, obs, env_output = env.step(action)
                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)
                            #landlord阵营赋值正的，farmer阵营赋值负的
                            episode_return = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
                            episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break

            for p in positions:
                while size[p] > T: 
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

def act_R_D(i, device, free_queue, full_queue, model, buffers, players, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['relation','dangerous']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        DEFAULT_CONFIG = {
            'allow_step_back': False,
            'seed': None,
        }

        env = Red_10Env(DEFAULT_CONFIG)
        env = Environment_R_D(env, device)

        done_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        player_id , obs_dict = env.initial()

        while True:
            rount_count = 0
            player_model = decide_model(env,players)
            while True:
                rount_count += 1
                for p in positions:
                    obs_x_buf[p].append(obs_dict['obs_'+p]['x'])
                    obs_z_buf[p].append(obs_dict['obs_'+p]['z'])
                    target_buf[p].append(obs_dict['obs_'+p]['target'])
                    size[p] += 1

                with torch.no_grad():
                    player_z = obs_dict['obs_player']['z_batch']
                    player_x = obs_dict['obs_player']['x_batch']
                    agent_output = player_model[env.env.game.round.current_player].forward(player_z, player_x)

                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs_dict['obs_player']['raw_legal_actions'][_action_idx]

                current_color = obs_dict['obs_' + p]['red_10_color']
                color = ''
                color_subindex = 0
                for action_card in action:
                    if action_card == 'T':
                        color += current_color[color_subindex]
                        color_subindex+=1

                obs_dict = env.step(action, color)
                if obs_dict['done']:
                    for sub_p in positions:
                        diff = size[sub_p]
                        if diff > 0:
                            done_buf[sub_p].extend([False for _ in range(diff - 1)])
                            done_buf[sub_p].append(True)
                            # landlord阵营赋值正的，farmer阵营赋值负
                    for r_i in range(rount_count):
                        target_buf['dangerous'][-(r_i+1)] = target_buf['dangerous'][-(r_i+1)]/float(rount_count)

                    break

            for p in positions:
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x'][index][t, ...] = obs_x_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_buf[p] = obs_x_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e

def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix

def _cards2tensor_R_D(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    matrix = _cards2array_R_D(list_cards)
    matrix = torch.from_numpy(matrix)
    return matrix