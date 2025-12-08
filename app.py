from simulated_annealing import get_figs, run_annealer
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

selected_frame_idx = st.number_input('enter frame id', value=210, step = 1)

latest_frame = game.tracking_data[game.tracking_data['frame']==selected_frame_idx].iloc[0]
annealer = Annealer(game, latest_frame)

st.session_state.new_game, best_score = run_annealer(annealer, game)
st.session_state.original, st.session_state.new_positions = get_figs(st.session_state.new_game, game, selected_frame_idx)

#streamlit
st.title(str(best_score))

@st.fragment
def figure_toggle():
    if st.button('rerun annealer'):
        st.session_state.new_game, best_score = run_annealer(annealer, game)
        st.session_state.original, st.session_state.new_positions = get_figs(st.session_state.new_game, game, selected_frame_idx)

    st.pyplot(st.session_state.new_positions[0] if st.toggle('see new positions') else st.session_state.original[0])

figure_toggle()

st.table(latest_frame)