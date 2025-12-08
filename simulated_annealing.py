import pandas as pd
from databallpy.features.pitch_control import get_pitch_control_single_frame
from databallpy.visualize import plot_soccer_pitch, plot_tracking_data
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy

from annealer import Annealer


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


def run_annealer(annealer: Annealer, game):
    best_solution, best_score = annealer.anneal() 
    new_game = deepcopy(game)
    new_game.tracking_data = pd.DataFrame(best_solution).transpose().reset_index()
    return new_game, best_score

def get_figs(new_game_obj, game, selected_frame_idx):
    original = generate_fig(game, selected_frame_idx)
    new_positions = generate_fig(new_game_obj, selected_frame_idx)
    return original, new_positions


