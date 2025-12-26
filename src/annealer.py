# Annealer
import random
import math
import pandas as pd
import numpy as np
from databallpy.features.pitch_control import get_team_influence
from databallpy.schemas.tracking_data import TrackingData
from copy import deepcopy
from databallpy import Game
from databallpy.utils.utils import sigmoid
from databallpy.utils.constants import DATABALLPY_POSITIONS

import pickle

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#note this is unreliable
NON_GK_POSITIONS = [p for p in DATABALLPY_POSITIONS if p != 'goalkeeper']

class Annealer:
    def __init__(self, game: Game, selected_frame_idx = 210, distance_perturbation = 0.2, num_iterations = 1000, max_distance_perturbation = 3, weighted_pitch_control_parameter = 1, pressing_parameter = 0, players_to_press = []):

        #data
        self.game = game
        self.frame = game.tracking_data[game.tracking_data['frame']==selected_frame_idx].iloc[0]

        self.attacking_team = self.frame['team_possession']
        self.defending_team = 'home' if self.frame['team_possession'] == 'away' else 'away'

        #annealing params
        self.distance_perturbation = distance_perturbation #how much to move the player
        self.p_0 = 0.5
        self.T_0 = -100/(math.log(self.p_0))
        self.T = self.T_0
        self.cooling_rate = 0.9    
        self.num_iterations = num_iterations
        self.max_distance_perturbation = max_distance_perturbation

        self.grid = np.meshgrid(
            np.linspace(-self.game.pitch_dimensions[0] / 2, self.game.pitch_dimensions[0] / 2, 106),
            np.linspace(-self.game.pitch_dimensions[1] / 2, self.game.pitch_dimensions[1] / 2, 68),
        )

        self.defending_player_ids = self.game.get_column_ids(team=self.defending_team, positions = NON_GK_POSITIONS)
        self.attacking_player_ids = self.game.get_column_ids(team=self.attacking_team, positions = NON_GK_POSITIONS)

        self.player_to_starting_position_map = self.frame[[c + '_x' for c in game.get_column_ids()] + [c + '_y' for c in game.get_column_ids()]]

        # since we are only moving defenders we can precompute this
        self.attacking_team_influence = get_team_influence(
        self.frame,
        col_ids=self.attacking_player_ids,
        grid=self.grid,
        player_ball_distances=None,
        )

        #define objective
        self.weighted_pitch_control_parameter = weighted_pitch_control_parameter
        self.pressing_parameter = pressing_parameter
        if self.pressing_parameter > 0:
            self.players_to_press = players_to_press if players_to_press else self.attacking_player_ids


        #compute matrix of importance of controlling each square in grid system
        with open(BASE_DIR / "xTArray.pkl", "rb") as f:
            self.xt_array = pickle.load(f)
        if self.attacking_team == 'away':
            self.xt_array = np.fliplr(self.xt_array)


    def perturbation(self, input_frame):
        new_frame = deepcopy(input_frame)
        #randomly choose 1 of the defenders
        #TODO see if we can cache the unselected players to avoid recomputing every time
        self.selected_players = random.sample(self.defending_player_ids, 1)
        for c in self.selected_players:
            #move each player up to a maximal distance from their starting positions
            x_col = c + '_x'
            y_col = c + '_y'
            player_initial_x_pos = self.player_to_starting_position_map[x_col]
            player_initial_y_pos = self.player_to_starting_position_map[y_col]
            new_x_pos = new_frame[x_col] + random.uniform(-self.distance_perturbation,self.distance_perturbation)
            new_y_pos = new_frame[y_col] + random.uniform(-self.distance_perturbation, self.distance_perturbation)

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
        #+ve is defending team, -ve is attacking team
        net_sigmoid_diff = sigmoid(team_influence_defending - self.attacking_team_influence, d=100) #this makes the sigmoid steeper and more binary

        defensive_matrix = self.xt_array * net_sigmoid_diff

        objective_matrix = defensive_matrix

        #pressure method only works on tracking data object, need to temporarily reconstruct it with the new data
        pressure_score = 0
        temp_tracking_df = pd.DataFrame(input_frame).T
        temp_tracking_df = TrackingData(temp_tracking_df.astype(self.game.tracking_data.dtypes))

        # add terms for tight marking by multiplying the objective by pressing parameter for controlling pitch near attacking player
        if self.pressing_parameter > 0:
            for attacking_player in self.players_to_press:
                pressure = temp_tracking_df.get_pressure_on_player(temp_tracking_df.index[0], attacking_player, [106,68], d_front = 9)
                #compute the mean pressure on a player by dividing by number of players in p
                pressure_score += pressure / len(self.players_to_press)
        
        #take the overall sum of geospatial terms
        return sum(sum(objective_matrix)) * self.weighted_pitch_control_parameter + pressure_score * self.pressing_parameter

    def anneal(self):
        best_score = 0
        best_solution = deepcopy(self.frame)
        latest_frame = deepcopy(self.frame)
        last_checkpoint_score = 0
        T = self.T
        print('running annealer')
        for i in range(1,self.num_iterations):
            perturbed_frame = self.perturbation(latest_frame)
            new_score = self.compute_objective(perturbed_frame)
            #take the new result if it's better, or randomly take a worse result with decaying probability
            if (new_score > best_score) or math.exp((new_score - best_score)/T) > random.random():
                latest_frame = perturbed_frame
            if (new_score > best_score):
                best_score = new_score
                best_solution = perturbed_frame
                latest_frame = perturbed_frame
            T = T*self.cooling_rate
            if i%100 == 0:
                print(f'iteration {i} / {self.num_iterations} best_score {best_score}')
                if last_checkpoint_score == best_score:
                    print('No improvement found in last 100 iterations... exiting.')
                    break
                last_checkpoint_score = best_score


        print(f'best score {best_score}')
        return best_solution, best_score




