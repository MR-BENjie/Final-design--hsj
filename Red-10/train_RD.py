import os

from douzero.dmc import parser, train_relation_danger

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train_relation_danger(flags)