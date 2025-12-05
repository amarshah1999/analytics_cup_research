from simulated_annealing import get_figs, run_annealer
import streamlit as st

original, new_positions = get_figs()

show_new_positions = st.toggle(label = 'show new positions')
if st.button('rerun annealer'):
    new_game = run_annealer()
    original, new_positions = get_figs(new_game)


if show_new_positions:
    st.pyplot(new_positions[0])
else:
    st.pyplot(original[0])
