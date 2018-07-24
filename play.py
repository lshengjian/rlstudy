#!/usr/bin/env python3
import numpy as np
import os,time
import imageio
from rlstudy.common import logger
from rlstudy.common.utils import make_atari_env, atari_arg_parser
from rlstudy.a2c.a2c import learn,Model
from rlstudy.a2c.policies import CnnPolicy
from rlstudy.common.atari_wrappers import make_atari, wrap_deepmind
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
START=223000
def make_model(env, total_timesteps=int(80e6),
              vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
              epsilon=1e-5, alpha=0.99):
    return Model(policy=CnnPolicy,ob_space=env.observation_space,
                  ac_space=env.action_space, nenvs=1,ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)
def play(**wrapper_kwargs):
    parser = atari_arg_parser()
    args=parser.parse_args()
    logger.configure()
    env = make_atari(args.env)
    env.seed(1234)
    env=wrap_deepmind(env, frame_stack=True,**wrapper_kwargs)
   
    model =make_model(env )
    model.load('models/'+args.env+'.save'+'.'+str(START))
    renders=[]

    obs = env.reset() #(84, 84, 4)
   
    for i in range(10000):
        
        obs = np.expand_dims(obs.__array__(), axis=0)#(1,84, 84, 4)
        a, v, _ ,_= model.step(obs)
        obs, reward, done, info = env.step(a[0])
        if i%10==0:
            img=env.render('rgb_array')
            renders.append(imageio.core.util.Image(img))
        else:
            env.render()
        if done:
            name = 'imgs/' + str(int(time.time())) + '.gif'
            imageio.mimsave(name, renders, duration=1 / 30)
            renders = []
            env.reset()

    env.close()

if __name__ == '__main__':
    play()
