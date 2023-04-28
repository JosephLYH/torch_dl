import os

from lib.agents import Battleship
from lib.envs import BattleshipEnv
from lib.metrics import MeanMetric

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
PLAYERS = 1
    
if __name__ == '__main__':
    env = BattleshipEnv(players=PLAYERS)
    agents = [Battleship(i, env.action_dim, env.action_dim) for i in range(PLAYERS)]
    mean_metrics = {
        'turns': MeanMetric(),
        'q': MeanMetric(),
        'loss': MeanMetric(),
        'reward': MeanMetric(),
    }

    episodes = 1000000
    for e in range(episodes):
        mean_metrics['loss'].reset()
        mean_metrics['q'].reset()
        mean_metrics['reward'].reset()

        state = env.random_init()

        # Play the game!
        while True:
            if agents[0].curr_step % 5000 == 0:
                print(f'episode {e}, step {env.turns}, agent {env.player_turn}')
                print(f'exploration rate: {agents[env.player_turn].exploration_rate}')
                env.render(env.player_turn)

            action = agents[env.player_turn].act(state)
            next_state, reward, done = env.step(env.player_turn, action)
            agents[env.player_turn].cache(state, next_state, action, reward, done)
            q, loss = agents[env.player_turn].learn()

            mean_metrics['reward'].update(reward)
            if q is not None and loss is not None:
                mean_metrics['loss'].update(loss)
                mean_metrics['q'].update(q)
            
            if done:
                mean_metrics['turns'].update(env.turns)
                metrics = {
                    'turns': env.turns, 
                    'mean_turns': mean_metrics['turns'].value(), 
                    'loss': mean_metrics['loss'].value(), 
                    'q': mean_metrics['q'].value(), 
                    'reward': mean_metrics['reward'].value()
                }
                agents[env.player_turn].write_tb(metrics)
                print(f'exploration rate: {agents[env.player_turn].exploration_rate}')
                print(f'episode {e}, step {env.turns}, agent {env.player_turn}')
                print(f"mean turns: {mean_metrics['turns'].value()}")
                print(f"mean reward: {mean_metrics['reward'].value()}")
                print(f"mean loss: {mean_metrics['loss'].value()}")
                print(f"mean q: {mean_metrics['q'].value()}")
                print()
                break

            state = env.get_next_turn_state()
            env.turns += 1