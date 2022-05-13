import multiprocessing as mp
import pickle

from douzero.env.game import GameEnv

#3个角色训练好的配置文件通过card_play_model_path_ditc传递给函数load_card_play_models，返回
#加载了相应配置文件的网络的DeepAgent，通过这个类的act函数，讲当前信息传递给网络或反馈一个最优的action，
#当前的信息构造通过game.py中的GameEnv的方法get_infoset构造的，其中的信息更新在GameEnv的step函数中
#在act中会根据传入的infoset的信息构造相应的obs传入网络作为输入。
def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ['landlord', 'landlord_up', 'landlord_down','landlord_front']:
        if card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        else:
            from .deep_agent import DeepAgent
            players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players


def load_card_play_models_R_10(model_dict , model_relation, model_dangerous):
    players = {str(i):{} for i in range(4)}
    from douzero.evaluation.deep_agent import _load_model
    for i in range(4):
        index = str(i)
        player_set = {}
        for position in ['landlord', 'landlord_up', 'landlord_down', 'landlord_front']:
            player_set[position] =_load_model(position, model_dict[position+"_"+index])
        players[index] = player_set

    from douzero.dmc.relation_decition import Relation_score_Model,Dangerous_Model
    import torch
    relation = Relation_score_Model()
    model_relation_state_dict = relation.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_relation, map_location='cuda:0')
    else:
        pretrained = torch.load(model_relation, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_relation_state_dict}
    model_relation_state_dict.update(pretrained)
    relation.load_state_dict(model_relation_state_dict)
    if torch.cuda.is_available():
        relation.cuda()
    relation.eval()

    danger = Dangerous_Model()
    model_danger_state_dict = danger.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_dangerous, map_location='cuda:0')
    else:
        pretrained = torch.load(model_dangerous, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_danger_state_dict}
    model_danger_state_dict.update(pretrained)
    danger.load_state_dict(model_danger_state_dict)
    if torch.cuda.is_available():
        danger.cuda()
    danger.eval()

    return players, relation , danger

def decide_players(relation_score, danger_score, players):
    model_numbles = [1000,1100,1010,1111]
    strnumber = ''
    countnumber = 0
    for x in relation_score:
        if x > danger_score:
            strnumber+= '1'
            countnumber+= 1
        else:
            strnumber+= '0'
    if countnumber == 0 :
        return players['0']['landlord'],model_numbles[0]
    elif countnumber == 1:
        if strnumber[0] == '1':
            return players['1']['landlord_down'],model_numbles[1]
        elif strnumber[1] == '1':
            return  players['2']['landlord'],model_numbles[2]
        elif strnumber[2] == '1':
            return players['1']['landlord'],model_numbles[1]
        else :
            raise Exception
    elif countnumber == 2:
        if strnumber[0] == '0':
            return players['0']['landlord_down'],model_numbles[0]
        elif strnumber[1] == '0':
            return  players['0']['landlord_front'],model_numbles[0]
        elif strnumber[2] == '0':
            return players['0']['landlord_up'],model_numbles[0]
        else :
            raise Exception
    elif countnumber == 3:
        if strnumber == '111':
            return players['3']['landlord'],model_numbles[3]
        else:
            raise  Exception

def mp_simulate(card_play_data_list, card_play_model_path_dict, q):

    players = load_card_play_models(card_play_model_path_dict)

    env = GameEnv(players)
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()

    q.put((env.num_wins['landlord'],
           env.num_wins['farmer'],
           env.num_scores['landlord'],
           env.num_scores['farmer']
         ))

def determinemember(env):
    import numpy as np
    landlords = env.env.game.state['landlord']
    strmode = np.zeros(4)
    countmode = len(landlords)
    for x in landlords:
        strmode[int(x)]=1
    if countmode == 1:
        finamode = 1000
    else:
        if strmode[0] == strmode[2]:
            finamode = 1010
        else :
            finamode = 1100
    return finamode, landlords


def mp_simulate_R_10(players ,model_relation,model_dangerous, q,num):
    from douzero.env.rlcard_red10.rlcard.envs.red_10 import Red_10Env
    from douzero.dmc.utils import Environment_R_D
    import torch

    try:
        DEFAULT_CONFIG = {
            'allow_step_back': False,
            'seed': None,
        }
        device = "0" if torch.cuda.is_available() else "cpu"
        env = Red_10Env(DEFAULT_CONFIG)
        env = Environment_R_D(env, device)

        wannermode = [1000]
        landlords = []
        while True:
            player_id, obs_dict = env.initial()
            mode, landlords= determinemember(env)
            if mode in wannermode:
                break

        buffer = dict()
        buffer['relation_score']= []
        buffer['relation_target']=[]
        buffer['dangerous_score']=[]
        buffer['dangerous_target']=[]
        buffer['red_10_mark']=[None ,None ]
        while True:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            relation_x = torch.unsqueeze(obs_dict['obs_relation']['x'],dim=0)
            relation_z = torch.unsqueeze(obs_dict['obs_relation']['z'],dim=0)
            buffer['relation_target'].append(obs_dict['obs_relation']['target'].cpu().numpy())
            relation_score = model_relation(relation_z, relation_x)

            dangerous_x = torch.unsqueeze(obs_dict['obs_dangerous']['x'],dim=0)
            dangerous_z = torch.unsqueeze(obs_dict['obs_dangerous']['z'],dim=0)
            buffer['dangerous_target'].append(obs_dict['obs_dangerous']['target'])
            dangerous_score = model_dangerous(dangerous_z,dangerous_x)

            relation_score = torch.squeeze(relation_score,dim=0).cpu().detach().numpy()
            dangerous_score = torch.squeeze(dangerous_score,dim=0).cpu().detach().numpy()
            buffer['relation_score'].append(relation_score)
            buffer['dangerous_score'].append(dangerous_score)


            player_model, model_number= decide_players(relation_score, dangerous_score, players)

            with torch.no_grad():
                player_z = obs_dict['obs_player']['z_batch']
                player_x = obs_dict['obs_player']['x_batch']
                agent_output = player_model.forward(player_z, player_x)
            _action_idx = int(agent_output['action'].cpu().detach().numpy())
            action = obs_dict['obs_player']['raw_legal_actions'][_action_idx]
            color = ''
            color_subindex = 0
            current_color = obs_dict['obs_relation']['red_10_color']
            for action_card in action:
                if action_card == 'T':
                    color += current_color[color_subindex]
                    color_subindex += 1


            if not buffer['red_10_mark'][0] and (
            env.env.game.played_colors[int(env.env.game.state['landlord'][0])][1:3]).sum() >= 1:
                buffer['red_10_mark'][0] = [int(env.env.game.state['landlord'][0]), obs_dict['obs_dangerous']['target']]
            """
            # get out if there are two red_10 players
            if not buffer['red_10_mark'][1] and (env.env.game.played_colors[env.env.game.state['landlord'][1]])[
                                                1:3].sum() == 1:
                buffer['red_10_mark'][1] = [int(env.env.game.state['landlord'][1]),
                                            obs_dict['obs_dangerous']['target']]
            """
            obs_dict = env.step(action, color)

            """
            for x in ['GAME : '+str(num)+'.\t',
                   "relation score : "+str(relation_score)+".\t",
                   "dangerous score : "+str(dangerous_score)+".\t",
                   "actions : "+str(action)+".\t",
                   "color : "+str(color )+".\t",
                   'players_mode : '+str(model_number)
                   ]:
                print(x ,end="")
            print("")
            
            q.put(('GAME : '+str(num)+'.\t',
                   "relation score : "+str(relation_score)+".\t",
                   "dangerous score : "+str(dangerous_score)+".\t",
                   "actions : "+str(action)+".\t",
                   "color : "+str(color )+".\t",
                   'players_mode : '+str(model_number)
                   ))
            """
            player_id = (player_id + 1) % 4
            if obs_dict['done']:
                print(str(num)+':end')
                break

        def compute_loss(logits, targets):
            loss = ((logits - targets) ** 2).mean()
            return loss
        import numpy as np
        length = len(buffer['dangerous_target'])
        buffer['dangerous_target'] = []
        for i in range(length):
            buffer['dangerous_target'].append((float(i+1)/length))
        re_dict = dict()
        all_relation_loss = compute_loss(np.array(buffer['relation_target']), np.array(buffer['relation_score']))
        all_dangerous_loss = compute_loss(
            np.array(buffer['dangerous_target']), np.array(buffer['dangerous_score']))

        re_dict["all_relation_loss"] = all_relation_loss
        re_dict['all_dangerous_loss'] = all_dangerous_loss
        re_dict['red_10_mark']=buffer['red_10_mark']
        re_dict['relation_score'] = buffer['relation_score']
        for p in landlords:
            p = int(p)
            t_1 = []
            s_1 = []
            for index, (target, score) in enumerate(zip(buffer['relation_target'],buffer['relation_score'])):
                if index %4 == p:
                    t_1.append(target)
                    s_1.append(score)
            re_dict['part_relation_loss_'+str(p)] = np.mean((np.array(t_1)-np.array(s_1))**2,axis = 0)
            re_dict['relation_score_'+str(p)] = s_1

        re_dict['landlords']=landlords
        return re_dict

    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e



def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker

def evaluate(landlord, landlord_up, landlord_down, landlord_front, eval_data, num_workers):

    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down,
        'landlord_front':landlord_front}

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_landlord_scores = 0
    num_farmer_scores = 0

    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_paly_data in card_play_data_list_each_worker:
        p = ctx.Process(
                target=mp_simulate,
                args=(card_paly_data, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_landlord_scores += result[2]
        num_farmer_scores += result[3]

    num_total_wins = num_landlord_wins + num_farmer_wins
    print('WP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('landlord : Farmers - {} : {}'.format(num_landlord_scores / num_total_wins, 2 * num_farmer_scores / num_total_wins)) 

def evaluate_R_10(model_dict, model_relation, model_dangerous, eval_num, num_workers):
    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    import numpy as np
    import visdom
    players, relation, dangerous = load_card_play_models_R_10(model_dict,model_relation,model_dangerous)
    loss_r=np.zeros(3,dtype=np.float)
    for index in range(eval_num):
        re_dict = mp_simulate_R_10(players, relation,dangerous, q, index)
        for p in re_dict['landlords']:
            loss_r += re_dict['part_relation_loss_'+str(p)]

        for p_id in range(4):
            vis = visdom.Visdom()
            if p_id in re_dict['landlords']:
                title = 'player '+str(p_id)+"(red)"
            else:
                title = 'player '+str(p_id)
            legend = ['up_p', 'front_p', 'down_p']
            y=[[0.,0.,0.]]
            if re_dict['red_10_mark'][0] is not None:
                legend.append('player ' + str(re_dict['red_10_mark'][0][0]) + " played red 10")
                y[0].append(0)
            if re_dict['red_10_mark'][1] is not None:
                legend.append('player ' + str(re_dict['red_10_mark'][1][0]) + " played red 10")
                y[0].append(0)
            vis.line(y,[0.],win='eval_'+str(p_id)+str(index),env = str(index), opts = dict(title = title,legend = legend))
            final_y = [0,0,0]
            final_round = [0,0,0]
            for i_s, x in enumerate(re_dict['relation_score']):
                index_sub = i_s % 4
                if index_sub == p_id:
                    vis.line([[float(x[0]), float(x[1]), float(x[2])]], [i_s], env = str(index),win='eval_'+str(p_id)+str(index), update='append')
                    for i in range(3):
                        final_y[i]=float(x[i])
                        final_round[i]=i_s

            if re_dict['red_10_mark'][0] is not None:
                y1 = final_y.copy()
                y2 = final_y.copy()
                y1.append(0)
                y2.append(1.5)
                final_round.append(re_dict['red_10_mark'][0][1])
                vis.line([y1,y2], [final_round,final_round], env = str(index),win='eval_' + str(p_id)+str(index),
                         update='append')
                vis.line([y1], [final_round],env = str(index),win='eval_' + str(p_id) + str(index),
                         update='append')
            if re_dict['red_10_mark'][0] is not None and re_dict['red_10_mark'][1] is not None:
                y1 = final_y.copy()
                y2 = final_y.copy()
                y1.extend([0,0])
                y2.extend([0,1.5])
                final_round.append(re_dict['red_10_mark'][1][1])
                vis.line([y1,y2], [final_round,final_round], env = str(index),win='eval_' + str(p_id)+str(index),
                         update='append')
            elif re_dict['red_10_mark'][1] is not None:
                y1 = final_y.copy()
                y2 = final_y.copy()
                y1.append(0)
                y2.append(1.5)
                final_round.append(re_dict['red_10_mark'][1][1])
                vis.line([y1,y2], [final_round, final_round],env = str(index),
                         win='eval_' + str(p_id) + str(index),
                         update='append')
        import time
        print(re_dict['red_10_mark'])