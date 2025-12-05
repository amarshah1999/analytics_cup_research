from simulated_annealing import get_figs
import streamlit as st

original, new_positions = get_figs()

show_new_positions = st.toggle(label = 'show new positions')
if show_new_positions:
    st.pyplot(new_positions[0])
else:
    st.pyplot(original[0])
