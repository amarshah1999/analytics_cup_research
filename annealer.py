# Annealer
import random
import math
from kloppy import skillcorner
import pandas as pd
import numpy as np
from databallpy import get_game_from_kloppy, get_saved_game
from databallpy.features.pitch_control import get_pitch_control_single_frame, get_team_influence
from databallpy.visualize import save_tracking_video, plot_soccer_pitch, plot_tracking_data
from kloppy import skillcorner, sportec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
from copy import deepcopy
from databallpy import Game

class Annealer:
    def __init__(self, game: Game, frame: pd.DataFrame):
        self.game = game
        self.frame = frame

        #annealing params
        self.tolerance = 0.5
        self.p_0 = 0.5
        self.T_0 = -100/(math.log(self.p_0))
        self.T = self.T_0
        self.cooling_rate = 0.9
            

        self.home_col_ids = [x[:-2] for x in self.frame.index if "home" in x and x[-2:] == "_x"]
        self.away_col_ids = [x[:-2] for x in self.frame.index if "away" in x and x[-2:] == "_x"]

        self.grid = np.meshgrid(
            np.linspace(-self.game.pitch_dimensions[0] / 2, self.game.pitch_dimensions[0] / 2, 106),
            np.linspace(-self.game.pitch_dimensions[1] / 2, self.game.pitch_dimensions[1] / 2, 68),
        )
    def perturbation(self, input_frame):
        #choose random player 
        #need to at some point make this work for home players
        new_frame = deepcopy(input_frame)
        for c in self.game.get_column_ids(team="away"):
            new_frame[c + '_x'] = new_frame[c + '_x'] + random.uniform(-self.tolerance,self.tolerance)
            new_frame[c + '_y'] = new_frame[c + '_y']+ random.uniform(-self.tolerance, self.tolerance)
        
        return new_frame

    def compute_objective(self, input_frame):
        team_influence_away = get_team_influence(
        input_frame,
        col_ids=self.away_col_ids,
        grid=self.grid,
        player_ball_distances=None,
        )
        # team_influence_home = get_team_influence(
        #     latest_frame,
        #     col_ids=home_col_ids,
        #     grid=grid,
        #     player_ball_distances=None,
        # )
        return sum(sum(team_influence_away))
    def anneal(self):
        best_score = 0
        best_solution = self.frame
        latest_frame = self.frame
        T = self.T
        
        for i in range(1,100):
            perturbed_frame = self.perturbation(latest_frame)
            new_score = self.compute_objective(perturbed_frame)
            if (new_score > best_score) or math.exp((new_score - best_score)/T) > random.random():
                latest_frame = perturbed_frame
            if (new_score > best_score):
                best_score = new_score
                best_solution = perturbed_frame
                latest_frame = perturbed_frame
            if i%10 == 0:
                print(best_score)
            T = T*self.cooling_rate

        return best_solution, best_score




