''' A toy example of self playing for red_10
'''
from rlcard.utils.utils import print_card
from rlcard.envs.red_10 import Red_10Env
# Make environment
num_players = 4
env = Red_10Env(config={'game_num_players': num_players})

print(">> red_10 human agent")
print(">> Start a new game")
game = env.game
game.init_game()
while (True):
    # If the human does not take the final action, we need to
    # print other players action

    # Let's take a look at what the agent card is
    current_state = game.state
    print_card(current_state[0]['state'][1])



    input("Press any key to continue...")
