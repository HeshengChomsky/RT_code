from DQN.DQN import train_DQN,evlution_DQN
import gym
from gym.envs.registration import register
register(
    id='BoxBall-v0',
    entry_point='Box_ball.grid_world:GridWorldEnv',
)

if __name__ == '__main__':
    # env = gym.make('BoxBall-v0')
    # train_DQN(env,env_name='BoxBall-v0',path='DQN/save_model')

    # env = gym.make('BoxBall-v0',render_mode="human")
    env = gym.make('BoxBall-v0')
    evlution_DQN(env, env_name='BoxBall-v0', path='save_model', epochs=50)