#3个角色训练好的配置文件通过card_play_model_path_ditc传递给函数load_card_play_models，返回
#加载了相应配置文件的网络的DeepAgent，通过这个类的act函数，讲当前信息传递给网络或反馈一个最优的action，
#当前的信息构造通过game.py中的GameEnv的方法get_infoset构造的，其中的信息更新在GameEnv的step函数中
#在act中会根据传入的infoset的信息构造相应的obs传入网络作为输入。
from douzero.evaluation.simulation import load_card_play_models
import douzero.evaluation.deep_agent as deep_agent
import douzero.env.game as Game
import douzero.env.env as Env
import torch
import numpy as np
#position is the sysmbol of the player(landlord...)
def getaction(position , model_path, infoset):
    player = deep_agent._load_model(position,model_path)
    if len(infoset.legal_actions) == 1:
        return infoset.legal_actions[0]

    obs = Env.get_obs(infoset)

    z_batch = torch.from_numpy(obs['z_batch']).float()
    x_batch = torch.from_numpy(obs['x_batch']).float()
    if torch.cuda.is_available():
        z_batch, x_batch = z_batch.cuda(), x_batch.cuda()

    y_pred = player.forward(z_batch, x_batch, return_value=True)['values']
    y_pred = y_pred.detach().cpu().numpy()

    best_action_index = np.argmax(y_pred, axis=0)[0]
    best_action = infoset.legal_actions[best_action_index]
    assert best_action in infoset.legal_actions
    return best_action

"""
    if len(action) > 0:
        self.last_pid = self.acting_player_position

    if action in bombs:
        self.bomb_num += 1

    self.last_move_dict[
        self.acting_player_position] = action.copy()

    self.card_play_action_seq.append(action)
    self.update_acting_player_hand_cards(action)

    self.played_cards[self.acting_player_position] += action

    if self.acting_player_position == 'landlord' and \
            len(action) > 0 and \
            len(self.three_landlord_cards) > 0:
        for card in action:
            if len(self.three_landlord_cards) > 0:
                if card in self.three_landlord_cards:
                    self.three_landlord_cards.remove(card)
            else:
                break

    self.game_done()
    if not self.game_over:
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()
    """