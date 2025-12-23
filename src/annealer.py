# Annealer
import random
import math
from re import T
import pandas as pd
import numpy as np
from databallpy.features.pitch_control import get_team_influence
from databallpy.schemas.tracking_data import TrackingData
from copy import deepcopy
from databallpy import Game
from databallpy.utils.utils import sigmoid
from databallpy.utils.constants import DATABALLPY_POSITIONS
from collections import defaultdict
import pickle
import accessible_space
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#note this is unreliable
non_gk_positions = [p for p in DATABALLPY_POSITIONS if p != 'goalkeeper']

class Annealer:
    def __init__(self, game: Game, selected_frame_idx = 210, distance_perturbation = 0.2, num_iterations = 1000, max_distance_perturbation = 3, weighted_pitch_control_parameter = 1, pressing_parameter = 10, passing_lane_block_parameter = 0, players_to_press = []):

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

        #define objective
        self.weighted_pitch_control_parameter = weighted_pitch_control_parameter
        self.pressing_parameter = pressing_parameter
        self.passing_lane_block_parameter = passing_lane_block_parameter
        self.players_to_press = players_to_press

        #FIXME need to improve this
        self.attacking_player_positions_on_grid = defaultdict(dict)
        temp = self.frame[[c + '_x' for c in self.attacking_player_ids] + [c + '_y' for c in self.attacking_player_ids]].apply(lambda x: math.floor(x))
        for c, coord in temp.items():
            #coordinate transform from 
            if c[-1] == 'x':
                self.attacking_player_positions_on_grid[c[:-2]]['x'] = int(coord + 106/2)
            else:
                self.attacking_player_positions_on_grid[c[:-2]]['y'] = int(coord + 68/2)

        self.ball_coords_in_grid_system = {'x': int(self.frame['ball_x'] + 106/2), 'y': int(self.frame['ball_y'] + 68/2) }

        #compute matrix of importance of controlling each square in grid system
        with open(BASE_DIR / "xTArray.pkl", "rb") as f:
            self.xt_array = pickle.load(f)
        if self.attacking_team == 'away':
            self.xt_array = np.fliplr(self.xt_array)
        

    # def get_dangerous_accessible_space_for_frame(self, frame):
    #     temp_game_for_accessible_space = deepcopy(self.game)
    #     temp_df = pd.DataFrame(frame).T
    #     temp_df = temp_df.astype(self.game.tracking_data.dtypes)
    #     temp_game_for_accessible_space.tracking_data = TrackingData(temp_df)
    #     df_long = temp_game_for_accessible_space.tracking_data.to_long_format()
    #     df_long["team"] = df_long["column_id"].str[:4]

    #     res = accessible_space.get_dangerous_accessible_space(
    #     df_long,
    #     frame_col="frame",
    #     period_col="period_id",
    #     player_col="column_id",
    #     team_col="team",
    #     x_col="x",
    #     y_col="y",
    #     vx_col="vx",
    #     vy_col="vy",
    #     team_in_possession_col="team_possession",
    #     player_in_possession_col="player_possession",
    #     use_progress_bar=False,
    #     )
    #     return res


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

    def get_grid_squares_between_points(self, x0, y0, x1, y1):
        """
        Efficiently computes all grid squares that lie between two points.
        Uses a supercover line algorithm to find all squares the line passes through.
        
        Args:
            x0, y0: Starting point (player position) as integer grid indices
            x1, y1: Ending point (ball position) as integer grid indices
            
        Returns:
            List of tuples (y, x) representing grid squares the line passes through
        """
        squares = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x = x0
        y = y0
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        # Add starting point
        squares.append((y, x))
        
        if dx > dy:
            # Horizontal movement dominates
            err = dx / 2.0
            while x != x1:
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                    squares.append((y, x))
                x += sx
                squares.append((y, x))
        else:
            # Vertical movement dominates
            err = dy / 2.0
            while y != y1:
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                    squares.append((y, x))
                y += sy
                squares.append((y, x))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_squares = []
        for square in squares:
            if square not in seen:
                seen.add(square)
                unique_squares.append(square)
        
        return unique_squares

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

        objective = defensive_matrix
       
        # defensive_interception_xt_array = np.fliplr(self.xt_array)

        # # defending_player_positions_on_grid = defaultdict(lambda: [None, None])
        # # passing_lanes_blocked = 0

        # # temp = input_frame[[c + '_x' for c in self.defending_player_ids] + [c + '_y' for c in self.defending_player_ids]].apply(lambda x: math.floor(x))
        # # for c, coord in temp.items():
        # #     #coordinate transform, where first item is y and second item is x in grid system
        # #     if c[-1] == 'x':
        # #         defending_player_positions_on_grid[c[:-2]][1] = int(coord + 106/2)
        # #     else:
        # #         defending_player_positions_on_grid[c[:-2]][0] = int(coord + 68/2)
        
        # # #can this be precomputed as a multiplier grid?
        # filtered_players_dict = {k: v for k, v in self.attacking_player_positions_on_grid.items() if k in self.players_to_press} if self.players_to_press else self.attacking_player_positions_on_grid

        # # add terms for tight marking by multiplying the objective by pressing parameter for controlling pitch near attacking player
        # if self.pressing_parameter != 0:
        #     for attacking_player, coords in filtered_players_dict.items():
        #         defensive_matrix[coords['y'], coords['x']] += self.pressing_parameter * self.pressing_importance_dict[attacking_player]

        # if self.passing_lane_block_parameter != 0:
        #     #add a term for blocking passing lanes - control pitch along line between player and ball
        #     for attacking_player, coords in filtered_players_dict.items():
        #         passing_lane = self.get_grid_squares_between_points(
        #             coords['x'], coords['y'],
        #             self.ball_coords_in_grid_system['x'], self.ball_coords_in_grid_system['y']
        #         )
        #         # if any(tuple[None, ...](c) in passing_lane for c in defending_player_positions_on_grid.values()):
        #         #     passing_lanes_blocked += self.passing_lane_block_parameter

        #         # Apply multiplier to all squares in the passing lane
        #         # passing_lane[0] is the attacking player's position, passing_lane[-1] is the ball's position
        #         # We taper it such that pressing the ball has value 0, pressing the player has value 1
        #         for i, (y, x) in enumerate(passing_lane):
        #             linear_taper_factor = (len(passing_lane) - i)/len(passing_lane)
        #             defensive_matrix[y, x] += self.passing_lane_block_parameter * defensive_interception_xt_array[y, x]

        # # # return sum(sum(team_influence_defending - self.attacking_team_influence))
        # # # return sum(sum(team_influence_defending)) + passing_lanes_blocked
        return sum(sum(objective))

    def anneal(self):
        best_score = 0

        best_solution = deepcopy(self.frame)
        latest_frame = deepcopy(self.frame)
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


        print(f'best score {best_score}')
        return best_solution, best_score




