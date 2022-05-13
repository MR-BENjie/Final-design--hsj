import os

import torch
import numpy as np
from collections import Counter

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12, 20: 13, 30: 14}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

def _get_one_hot_bomb(bomb_num):
    one_hot = np.zeros(15, dtype=np.float32)
    one_hot[bomb_num] = 1
    return one_hot

def _load_model(position, model_dir, use_onnx):
    if not use_onnx or not os.path.isfile(os.path.join(model_dir, position+'.onnx')) :
        from models import model_dict
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

def _process_action_seq(sequence, length=15):
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

class DeepAgent:

    def __init__(self, position, model_dir, use_onnx=False):
        self.model = _load_model(position, model_dir, use_onnx)
        self.use_onnx = use_onnx

    def cards2array(self, list_cards):
        if len(list_cards) == 0:
            return np.zeros(54, dtype=np.float32)

        matrix = np.zeros([4, 13], dtype=np.float32)
        jokers = np.zeros(2, dtype=np.float32)
        counter = Counter(list_cards)
        for card, num_times in counter.items():
            if card < 20:
                matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
            elif card == 20:
                jokers[0] = 1
            elif card == 30:
                jokers[1] = 1
        return np.concatenate((matrix.flatten('F'), jokers))

    def get_one_hot_array(self, num_left_cards, max_num_cards):
        one_hot = np.zeros(max_num_cards, dtype=np.float32)
        one_hot[num_left_cards - 1] = 1

        return one_hot

    def action_seq_list2array(self, action_seq_list):
        action_seq_array = np.zeros((len(action_seq_list), 54), dtype=np.float32)
        for row, list_cards in enumerate(action_seq_list):
            action_seq_array[row, :] = self.cards2array(list_cards)
        action_seq_array = action_seq_array.reshape(5, 216)
        return action_seq_array

    def act(self, infoset):
        player_position = infoset.player_position
        num_legal_actions = len(infoset.legal_actions)

        my_handcards = self.cards2array(infoset.player_hand_cards)
        my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                       num_legal_actions, axis=0)

        other_handcards = self.cards2array(infoset.other_hand_cards)
        other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                          num_legal_actions, axis=0)

        my_action_batch = np.zeros(my_handcards_batch.shape, dtype=np.float32)
        for j, action in enumerate(infoset.legal_actions):
            my_action_batch[j, :] = self.cards2array(action)

        last_action = self.cards2array(infoset.rival_move)
        last_action_batch = np.repeat(last_action[np.newaxis, :],
                                      num_legal_actions, axis=0)

        p_id = infoset.player_position
        teammate_u_id = (p_id - 1) % 4
        teammate_u_played_cards = self.cards2array(infoset.played_cards[teammate_u_id])
        teammate_u_played_cards_batch = np.repeat(
            teammate_u_played_cards[np.newaxis, :],
            num_legal_actions, axis=0)

        last_teammate_u_action = self.cards2array(infoset.last_moves[teammate_u_id])
        last_teammate_u_action_batch = np.repeat(
            last_teammate_u_action[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_u_num_cards_left = self.get_one_hot_array(infoset.num_cards_left[teammate_u_id], 13)
        teammate_u_num_cards_left_batch = np.repeat(
            teammate_u_num_cards_left[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_f_id = (p_id - 2) % 4
        teammate_f_played_cards = self.cards2array(infoset.played_cards[teammate_f_id])
        teammate_f_played_cards_batch = np.repeat(
            teammate_f_played_cards[np.newaxis, :],
            num_legal_actions, axis=0)

        last_teammate_f_action = self.cards2array(infoset.last_moves[teammate_f_id])
        last_teammate_f_action_batch = np.repeat(
            last_teammate_f_action[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_f_num_cards_left = self.get_one_hot_array(infoset.num_cards_left[teammate_f_id], 13)
        teammate_f_num_cards_left_batch = np.repeat(
            teammate_f_num_cards_left[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_d_id = (p_id - 3) % 4
        teammate_d_played_cards = self.cards2array(infoset.played_cards[teammate_d_id])
        teammate_d_played_cards_batch = np.repeat(
            teammate_d_played_cards[np.newaxis, :],
            num_legal_actions, axis=0)

        last_teammate_d_action = self.cards2array(infoset.last_moves[teammate_d_id])
        last_teammate_d_action_batch = np.repeat(
            last_teammate_d_action[np.newaxis, :],
            num_legal_actions, axis=0)

        teammate_d_num_cards_left = self.get_one_hot_array(infoset.num_cards_left[teammate_d_id], 13)
        teammate_d_num_cards_left_batch = np.repeat(
            teammate_d_num_cards_left[np.newaxis, :],
            num_legal_actions, axis=0)

        bomb_num = self.get_one_hot_array(1, 15)
        bomb_num_batch = np.repeat(
            bomb_num[np.newaxis, :],
            num_legal_actions, axis=0)
        x_batch = np.hstack((my_handcards_batch,
                             other_handcards_batch,
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
        z = self.action_seq_list2array(_process_action_seq(infoset.card_play_action_seq,length=20))
        z_batch = np.repeat(
            z[np.newaxis, :, :],
            num_legal_actions, axis=0)
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

        #best_action_index = np.argmax(y_pred, axis=0)[0]
        size = min(3, len(y_pred))
        best_action_index = np.argpartition(y_pred, -size)[-size:]
        best_action_confidence = y_pred[best_action_index]
        best_action = [infoset.legal_actions[index] for index in best_action_index]

        return best_action, best_action_confidence
