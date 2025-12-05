# Annealer
from typing import Any
import random
import math
from kloppy import skillcorner
import pandas as pd
import numpy as np
from databallpy import get_game_from_kloppy, get_saved_game
from databallpy.features.pitch_control import get_pitch_control_single_frame
from databallpy.visualize import save_tracking_video, plot_soccer_pitch, plot_tracking_data
from kloppy import skillcorner, sportec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
from copy import deepcopy
from annealer import Annealer

match_id = 1886347
frame_idx = 210

tracking_data_github_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json"

dataset = skillcorner.load(
    meta_data=meta_data_github_url,
    raw_data=tracking_data_github_url,
    # Optional Parameters
    coordinates="skillcorner",  # or specify a different coordinate system
    # sample_rate=(1 / 2),  # changes the data from 10fps to 5fps
    limit=1000,  # only load the first 100 frames
)

def generate_fig(game, frame_idx):
    frame = game.tracking_data[game.tracking_data['frame']==frame_idx].iloc[0]
    pitch_control = get_pitch_control_single_frame(frame,
            game.pitch_dimensions,
            n_x_bins=106, n_y_bins=68,
        )
    cmap_red_green = LinearSegmentedColormap.from_list("reds", [(0, 1, 0, 1), (0.5, 0.5, 0, 0), (1, 0, 0, 1)])

    fig, ax = plot_soccer_pitch(field_dimen=game.pitch_dimensions, pitch_color="white")
    idx_relative_to_game = game.tracking_data[game.tracking_data['frame']==frame_idx].index[0]
    fig, ax = plot_tracking_data(
    game,
    idx_relative_to_game,
    team_colors=["green", "red"],
    ax=ax,
    fig=fig,
    heatmap_overlay=pitch_control,
    overlay_cmap=cmap_red_green,
    add_velocities = True
    )
    return fig,ax

#actual script
game = get_game_from_kloppy(dataset)
game.tracking_data.add_velocity(game.get_column_ids() + ["ball"], allow_overwrite=True)


latest_frame = game.tracking_data[game.tracking_data['frame']==frame_idx].iloc[0]

best_solution, best_score = Annealer(game, latest_frame).anneal() 
new_game = deepcopy(game)
new_game.tracking_data = pd.DataFrame(best_solution).transpose().reset_index()
def get_figs():
    original = generate_fig(game, frame_idx)
    new_positions = generate_fig(new_game, frame_idx)
    return original, new_positions


