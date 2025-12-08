# Annealer
import random
import math
import pandas as pd
import numpy as np
from databallpy.features.pitch_control import get_team_influence, get_player_influence
from copy import deepcopy
from databallpy import Game
from databallpy.utils.utils import sigmoid
from databallpy.utils.constants import DATABALLPY_POSITIONS
from collections import defaultdict
#note this is unreliable
non_gk_positions = [p for p in DATABALLPY_POSITIONS if p != 'goalkeeper']
class Annealer:
    def __init__(self, game: Game, frame: pd.DataFrame):
        #data
        self.game = game
        self.frame = frame

        self.attacking_team = frame['team_possession']
        self.defending_team = 'home' if frame['team_possession'] == 'away' else 'away'

        #annealing params
        self.tolerance = 0.2
        self.p_0 = 0.2
        self.T_0 = -100/(math.log(self.p_0))
        self.T = self.T_0
        self.cooling_rate = 0.9    
        self.num_iterations = 1000
        self.max_distance_perturbation = 3

        self.grid = np.meshgrid(
            np.linspace(-self.game.pitch_dimensions[0] / 2, self.game.pitch_dimensions[0] / 2, 106),
            np.linspace(-self.game.pitch_dimensions[1] / 2, self.game.pitch_dimensions[1] / 2, 68),
        )

        self.defending_player_ids = self.game.get_column_ids(team=self.defending_team, positions = non_gk_positions)
        self.attacking_player_ids = self.game.get_column_ids(team=self.attacking_team, positions = non_gk_positions)

        self.player_to_starting_position_map = self.frame[[c + '_x' for c in game.get_column_ids()] + [c + '_y' for c in game.get_column_ids()]]

        # since we are only moving defenders we can precompute this
        self.attacking_team_influence = get_team_influence(
        self.frame,
        col_ids=self.attacking_player_ids,
        grid=self.grid,
        player_ball_distances=None,
        )
        
        #FIXME need to improve this
        self.attacking_player_positions_on_grid = defaultdict(dict)
        temp = self.frame[[c + '_x' for c in self.attacking_player_ids] + [c + '_y' for c in self.attacking_player_ids]].apply(lambda x: math.floor(x))
        for c, coord in temp.items():
            if c[-1] == 'x':
                self.attacking_player_positions_on_grid[c[:-2]]['x'] = int(coord + 106/2)
            else:
                self.attacking_player_positions_on_grid[c[:-2]]['y'] = int(coord + 34/2)

    def perturbation(self, input_frame):
        new_frame = deepcopy(input_frame)
        #randomly choose 6 of the defenders
        #TODO see if we can cache the unselected players to avoid recomputing every time
        self.selected_players = random.sample(self.defending_player_ids, 3)
        for c in self.selected_players:
            #move each player up to a maximal distance from their starting positions
            x_col = c + '_x'
            y_col = c + '_y'
            player_initial_x_pos = self.player_to_starting_position_map[x_col]
            player_initial_y_pos = self.player_to_starting_position_map[y_col]
            new_x_pos = new_frame[x_col] + random.uniform(-self.tolerance,self.tolerance)
            new_y_pos = new_frame[y_col] + random.uniform(-self.tolerance, self.tolerance)

            if player_initial_x_pos - self.max_distance_perturbation < new_x_pos < player_initial_x_pos + self.max_distance_perturbation:
                new_frame[x_col] = new_x_pos
            if player_initial_y_pos - self.max_distance_perturbation < new_y_pos < player_initial_y_pos + self.max_distance_perturbation:
                new_frame[y_col] = new_y_pos

        
        return new_frame

    def compute_objective(self, input_frame):

        team_influence_defending = get_team_influence(
        input_frame,
        col_ids=self.defending_player_ids,
        grid=self.grid,
        player_ball_distances=None,
        )

        #add in extra arbitrary terms
        # player_influence = []
        # for col_id in self.defending_player_ids:
        #     distance_to_ball = np.linalg.norm(
            #     frame[[f"{col_id}_x", f"{col_id}_y"]].values
            #     - frame[["ball_x", "ball_y"]].values
            # )

        #     player_influence.append(
        #         get_player_influence(
        #             x_val=input_frame[f"{col_id}_x"],
        #             y_val=input_frame[f"{col_id}_y"],
        #             vx_val=input_frame[f"{col_id}_vx"],
        #             vy_val=input_frame[f"{col_id}_vy"],
        #             distance_to_ball=distance_to_ball,
        #             grid=self.grid,
        #         )
        #     )
        
        #add terms for tight marking by multiplying the objective by 1000 for controlling pitch near attacking player
        for attacking_player, coords in self.attacking_player_positions_on_grid.items():
            #TODO add a multiplier so that different players can be overweighted
            team_influence_defending[coords['y'], coords['x']] *= 10

        return sum(sum(sigmoid(team_influence_defending - self.attacking_team_influence)))
        # return sum(sum(team_influence_defending))

    def anneal(self):
        best_score = 0
        best_solution = self.frame
        latest_frame = self.frame
        T = self.T
        print('running annealer')
        for i in range(1,self.num_iterations):
            perturbed_frame = self.perturbation(latest_frame)
            new_score = self.compute_objective(perturbed_frame)
            if (new_score > best_score) or math.exp((new_score - best_score)/T) > random.random():
                latest_frame = perturbed_frame
            if (new_score > best_score):
                best_score = new_score
                best_solution = perturbed_frame
                latest_frame = perturbed_frame
            T = T*self.cooling_rate
            if i%100 == 0:
                print(f'best_score {best_score}')

        print(f'best score {best_score}')
        return best_solution, best_score




