#!/usr/bin/env python3
import numpy as np
import os
from rlstudy.common import logger
from rlstudy.common.utils import make_atari_env, atari_arg_parser
from rlstudy.common.vec_env.vec_frame_stack import VecFrameStack
from rlstudy.a2c.a2c import learn,Model
from rlstudy.a2c.policies import CnnPolicy
from rlstudy.common.atari_wrappers import make_atari, wrap_deepmind

START=223000
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train(env_id, num_timesteps, seed,  num_env):
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    learn(env_id,env, CnnPolicy, seed, total_timesteps=int(num_timesteps * 1.1),start=START)
    env.close()

def main():
    parser = atari_arg_parser()
    #parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    #parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    logger.configure(dir='logs')
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, num_env=8)


if __name__ == '__main__':
    main()
