import streamlit as st
import pandas as pd
import numpy as np
from databallpy import get_game_from_kloppy, get_saved_game
from databallpy.features.pitch_control import get_pitch_control_single_frame
from databallpy.visualize import save_tracking_video, plot_soccer_pitch, plot_tracking_data
from kloppy import skillcorner, sportec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import polars as pl

#constants and defs
match_id = 1886347
@st.cache_data
def load_event_data(github_url):
    return pl.read_csv(github_url)

def get_game():
     return get_saved_game(f"{match_id}_game.pkl")



# game.save_game(f"{match_id}_game.pkl")
event_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_dynamic_events.csv"
st.title(f'Event data for match {match_id}')
data = load_event_data(event_data_github_url)
event_data = st.write(data)




# ------------------=================
# Get the selected row index
st.title('Tracking data')
game = get_game()
cleaned_tracking_data = game.tracking_data[~game.tracking_data.datetime.isna()].reset_index()
tracking_data_table = st.write(cleaned_tracking_data)

selected_frame = st.slider('Frame', 0, len(cleaned_tracking_data)-1)

pitch_control = get_pitch_control_single_frame(cleaned_tracking_data.iloc[selected_frame],
    game.pitch_dimensions,
    n_x_bins=105, n_y_bins=68,
)
cmap_red_green = LinearSegmentedColormap.from_list("reds", [(0, 1, 0, 1), (0.5, 0.5, 0, 0), (1, 0, 0, 1)])

fig, ax = plot_soccer_pitch(field_dimen=game.pitch_dimensions, pitch_color="white")
fig, ax = plot_tracking_data(game, selected_frame, team_colors=["green", "red"], ax=ax, fig=fig, heatmap_overlay=pitch_control, overlay_cmap=cmap_red_green)
st.pyplot(fig)

