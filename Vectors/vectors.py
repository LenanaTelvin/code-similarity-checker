import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

mode1 =SentenceTransformer('all-MiniLM-L6-v2',device='cpu')
_ = mode1.encode(["test"], show_progress_bar=False)
#function to compute code similarity

def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

st.title("Code Similarity Checker")
st.write("Enter two code snippets to check their similarity.")  

code1 = st.text_area("paste code Snippet 1",height=200)
code2 = st.text_area("paste code Snippet 2",height=200)

if st.button("Check Similarity"):
    if code1.strip() and code2.strip():
        vec1 = mode1.encode(code1)
        vec2 = mode1.encode(code2)
        similarity = cosine_similarity(vec1, vec2)
        st.write(f"Similarity Score: {similarity:.4f}")
        
        if similarity > 0.8:
            st.info("The code snippets are highly similar.")
        elif similarity > 0.5:
            st.warning("The code snippets have moderate similarity.")
        else:
            st.error("The code snippets are not similar.")
    else:
        st.error("Please enter both code snippets to check similarity.")