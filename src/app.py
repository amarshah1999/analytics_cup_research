from simulated_annealing import generate_fig, run_annealer
import streamlit as st
from kloppy import skillcorner
from databallpy import get_game_from_kloppy
from annealer import Annealer
import pandas as pd

match_id = 1886347
tracking_data_github_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json"

dataset = skillcorner.load(
    meta_data=meta_data_github_url,
    raw_data=tracking_data_github_url,
    # Optional Parameters
    coordinates="skillcorner",  # or specify a different coordinate system
    # sample_rate=(1 / 2),  # changes the data from 10fps to 5fps
    limit=2000,  # only load the first 100 frames
)


#actual script
game = get_game_from_kloppy(dataset)
game.tracking_data.add_velocity(game.get_column_ids() + ["ball"], allow_overwrite=True)
game.tracking_data.add_individual_player_possession()
selected_frame_idx = st.number_input('enter frame id', value=210, step = 1)



# Get available player IDs from the tracking data

player_columns = game.get_column_ids(team=game.tracking_data[game.tracking_data['frame']==selected_frame_idx].iloc[0]['team_possession'])


st.session_state.original = generate_fig(game, selected_frame_idx)


@st.fragment
def figure_toggle():
    distance_perturbation = st.number_input('distance perturbation', 0.0, 3.0, value=0.5)
    num_iterations = st.number_input('num iterations', value=2000)
    max_distance_perturbation = st.number_input('max distance perturbation', value = 3)
    pressing_parameter = st.number_input('pressing parameter', value=0.5)
    players_to_press = st.multiselect(
        'Select players to press',
        options=player_columns,
        default=[]
    )
    if st.button('run annealer'):
        annealer = Annealer(
            game, 
            selected_frame_idx, 
            distance_perturbation=distance_perturbation, 
            max_distance_perturbation=max_distance_perturbation, 
            num_iterations=num_iterations,
            pressing_parameter = pressing_parameter,
            players_to_press = players_to_press
        )
        st.session_state.new_game, best_score = run_annealer(annealer, game)
        st.session_state.new_positions = generate_fig(st.session_state.new_game, selected_frame_idx)
    if 'new_positions' in st.session_state:
        graph = st.pyplot(st.session_state.new_positions[0] if st.toggle('see new positions') else st.session_state.original[0])
    else:
        graph = st.pyplot(st.session_state.original[0])

figure_toggle()

# st.table(latest_frame)