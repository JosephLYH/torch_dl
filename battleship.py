import os

from lib.agents import Battleship
from lib.envs import BattleshipEnv
from lib.metrics import MeanMetric

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
PLAYERS = 1
    
if __name__ == '__main__':
    env = BattleshipEnv(players=PLAYERS)
    agents = [Battleship(i, env.action_dim, env.action_dim) for i in range(PLAYERS)]
    mean_turns = MeanMetric()

    episodes = 1000000
    for e in range(episodes):
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

            if done:
                mean_turns.update(env.turns)
                print(f'exploration rate: {agents[env.player_turn].exploration_rate}')
                print(f'episode {e}, step {env.turns}, agent {env.player_turn}')
                print(f'Mean turns: {mean_turns.mean()}')
                print()
                break

            state = env.get_next_turn_state()
            env.turns += 1