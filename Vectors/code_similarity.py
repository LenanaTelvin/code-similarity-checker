import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns

# default embedding model
mode1 ={"default": SentenceTransformer('all-MiniLM-L6-v2', device='cpu')}

#supported file extensions 
LANGUAGE_MAP = {
    ".py" : "Python",".java": "Java", ".js": "Javascript",".cpp": "C++", ".c": "C", ".cs": "C#",
    ".ts": "Typescript", ".rb": "Ruby", ".go": "Go", ".php": "PHP", ".swift": "Swift"
}

# Function to compute cosine similarity
def cosine_similarity(a,b):
    a =np.array(a)
    b =np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# APP (UI) user interace
st.title ("Code Similarity Checker with file upload with Language detection")
st.write("Enter two code snippets or upload files to check their similarity.")

# File uploader this allows multiple file uploads matching our supported extensions
uploaded_files = st.file_uploader(
    "Upload code files", type=list(LANGUAGE_MAP.keys()), accept_multiple_files=True
)

if uploaded_files:
    snippets = []
    for uploaded_file in uploaded_files:
        # Read the content of uploaded file
        content =uploaded_file.read().decode("utf-8",errors='ignore')
        ext =os.path.splitext(uploaded_file.name)[1]
        lang =LANGUAGE_MAP.get(ext, "Unknown")
        snippets.append({
            "name": uploaded_file.name,
            "language": lang,
            "content": content
        })

    df_meta =pd.DataFrame([{'File' :s['name'],'Language' : s['language']} for s in snippets])
    st.table(df_meta)

    # Compute embeddings for each code snippet using the default model
    for s in snippets:
        # The encode() method turns code text into a numeric vector (embedding)
        s['vec'] = mode1['default'].encode(s['content'])

    if len(snippets) ==2:
        vec1, vec2 = snippets[0]['vec'], snippets[1]['vec']
        similarity = cosine_similarity(vec1,vec2)
        st.write(f"Similarity Score: {similarity :4f}")

        st.metric(f"Similarity{ snippets[0]['name']} and {snippets[1]['name']}",  value =f"{similarity:.4f}")
   
    else:
        name =[s['name'] for s in snippets]
        mat =np.zeros((len(snippets), len(snippets)))
        for i in range(len(snippets)):
            for j in range(len(snippets)):
                if i != j:
                    mat[i][j] = cosine_similarity(snippets[i]['vec'],snippets[j]['vec'])
        df_mat = pd.DataFrame(mat, index=name, columns=name)
        st.table(df_mat)
        st.write("Similarity Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_mat, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, ax=ax)
        ax.set_title("Code Similarity Heatmap", fontsize=14)
        st.pyplot(fig)

else:
    st.warning("Please upload at least two code files to check similarity.")