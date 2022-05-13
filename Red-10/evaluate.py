import os 
import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')

    parser.add_argument('--landlord_', type=str,
            default='baselines/douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_up_', type=str,
            default='baselines/sl/landlord_up.ckpt')
    parser.add_argument('--landlord_down_', type=str,
            default='baselines/sl/landlord_down.ckpt')
    parser.add_argument('--landlord_front_', type=str,
                        default='baselines/sl/landlord_down.ckpt')


    parser.add_argument('--relation', type=str,
                        default='baselines/relation.ckpt')
    parser.add_argument('--dangerous' , type=str,
                        default='baselines/dangerous.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.landlord_front,
             args.eval_data,
             args.num_workers)
