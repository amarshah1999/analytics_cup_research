import streamlit as st
import pandas as pd
import numpy as np
from databallpy import get_game_from_kloppy, get_saved_game
from databallpy.features.pitch_control import get_pitch_control_single_frame
from databallpy.visualize import save_tracking_video, plot_soccer_pitch, plot_tracking_data
from kloppy import skillcorner, sportec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

matches = pd.read_json('https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches.json')

#constants and defs
match_id = st.selectbox(
    "",
    list(matches.id.unique()),
    index=9,
    placeholder="Select match",
)
@st.cache_data
def load_event_data(github_url) -> pd.DataFrame():
    return pd.read_csv(github_url)

def get_game():
    try:
        g = get_saved_game(f"{match_id}_game.pkl")
        #get_column_ids returns player positions but doesn't return the ball
        g.tracking_data.add_velocity(g.get_column_ids() + ["ball"], allow_overwrite=True)
        return g
    except ValueError:
        tracking_data_github_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
        meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json"
        tracking_dataset = skillcorner.load(
            meta_data=meta_data_github_url,
            raw_data=tracking_data_github_url,
            # Optional Parameters
            coordinates="skillcorner",  # or specify a different coordinate system
        )
        game = get_game_from_kloppy(tracking_dataset)
        game.save_game(f"{match_id}_game.pkl", allow_overwrite=True)
        game.tracking_data.add_velocity(game.get_column_ids() + ["ball"], allow_overwrite=True)
        return game

     




game = get_game()

event_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches/{match_id}/{match_id}_dynamic_events.csv"
st.title(f'Event data for match {match_id}')
data = load_event_data(event_data_github_url)
event_data_table = st.dataframe(
    data,
    selection_mode = "multi-row",
    on_select = "rerun"
    )

if event_data_table.selection.rows:
    frame_start = data.iloc[event_data_table.selection.rows[0]].frame_start
    frame_end = data.iloc[event_data_table.selection.rows[-1]].frame_end

# # ------------------=================
# # Get the selected row index
    tracking_columns = [c + '_x' for c in game.get_column_ids()] + [c + '_y' for c in game.get_column_ids()]

    cleaned_tracking_data = game.tracking_data[~game.tracking_data.datetime.isna()].reset_index()
    cleaned_tracking_data[tracking_columns] = cleaned_tracking_data[tracking_columns].interpolate(method = 'linear')
    cleaned_tracking_data = cleaned_tracking_data[(cleaned_tracking_data.frame >= frame_start)&(cleaned_tracking_data.frame <= frame_end)]
    tracking_data_table = st.write(cleaned_tracking_data)

    if len(cleaned_tracking_data) > 0:
        selected_frame_relative_to_event = st.slider('Frame', 0, len(cleaned_tracking_data)-1)
        selected_frame_relative_to_game = cleaned_tracking_data.iloc[selected_frame_relative_to_event].frame
        st.title((frame_start, frame_end, selected_frame_relative_to_game))
        pitch_control = get_pitch_control_single_frame(cleaned_tracking_data.iloc[selected_frame_relative_to_event],
            game.pitch_dimensions,
            n_x_bins=105, n_y_bins=68,
        )
        
        cmap_red_green = LinearSegmentedColormap.from_list("reds", [(0, 1, 0, 1), (0.5, 0.5, 0, 0), (1, 0, 0, 1)])

        fig, ax = plot_soccer_pitch(field_dimen=game.pitch_dimensions, pitch_color="white")
        fig, ax = plot_tracking_data(
            game
        , selected_frame_relative_to_game
        , team_colors=["green", "red"]
        , ax=ax
        , fig=fig
        , heatmap_overlay=pitch_control
        , overlay_cmap=cmap_red_green
        , add_velocities = True
        
        )
        st.pyplot(fig)


