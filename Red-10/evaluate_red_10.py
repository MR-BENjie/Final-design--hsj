import os
import argparse

from douzero.evaluation.simulation import evaluate_R_10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Dou Dizhu Evaluation')
    model_numbles = [1000, 1100, 1010, 1111]
    # _0 :1000 ; _1:1100; _2:1010; _3:1111
    for i in range(4):
        parser.add_argument('--landlord_' + str(i), type=str,
                            default='douzero_checkpoints/douzero_'+str(model_numbles[i])+'/model/landlord_weights.ckpt')
        parser.add_argument('--landlord_up_' + str(i), type=str,
                            default='douzero_checkpoints/douzero_'+str(model_numbles[i])+'/model/landlord_up_weights.ckpt')
        parser.add_argument('--landlord_down_' + str(i), type=str,
                            default='douzero_checkpoints/douzero_'+str(model_numbles[i])+'/model/landlord_down_weights.ckpt')
        parser.add_argument('--landlord_front_' + str(i), type=str,
                            default='douzero_checkpoints/douzero_'+str(model_numbles[i])+'/model/landlord_front_weights.ckpt')

    parser.add_argument('--relation', type=str,
                        default='R_D_checkpoints/relation_weights.ckpt')
    parser.add_argument('--dangerous', type=str,
                        default='R_D_checkpoints/dangerous_weights.ckpt')
    parser.add_argument('--eval_num', type=int,
                        default=10)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device


    mode_weight = dict()
    mode_weight["landlord_0"] = args.landlord_0
    mode_weight["landlord_up_0"] = args.landlord_up_0
    mode_weight["landlord_front_0"] = args.landlord_front_0
    mode_weight["landlord_down_0"] = args.landlord_down_0

    mode_weight["landlord_1"] = args.landlord_1
    mode_weight["landlord_up_1"] = args.landlord_up_1
    mode_weight["landlord_front_1"] = args.landlord_front_1
    mode_weight["landlord_down_1"] = args.landlord_down_1

    mode_weight["landlord_2"] = args.landlord_2
    mode_weight["landlord_up_2"] = args.landlord_up_2
    mode_weight["landlord_front_2"] = args.landlord_front_2
    mode_weight["landlord_down_2"] = args.landlord_down_2

    mode_weight["landlord_3"] = args.landlord_3
    mode_weight["landlord_up_3"] = args.landlord_up_3
    mode_weight["landlord_front_3"] = args.landlord_front_3
    mode_weight["landlord_down_3"] = args.landlord_down_3

    evaluate_R_10(mode_weight,
             args.relation,
             args.dangerous,
             args.eval_num,
             args.num_workers)
