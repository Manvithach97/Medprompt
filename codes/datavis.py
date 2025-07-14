import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import Counter
import tempfile

def save_entity_distribution_chart(entities):
    entity_types = [e[1] for e in entities]
    counts = Counter(entity_types)

    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values(), color="skyblue")
    ax.set_title("Entity Type Distribution")
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmpfile.name, bbox_inches="tight")
    plt.close(fig)

    return tmpfile.name

def plot_entity_distribution(entities):
    if not entities:
        st.warning("No entities to visualize.")
        return

    df = pd.DataFrame(entities, columns=["Entity", "Entity Type"])
    entity_counts = df["Entity Type"].value_counts().reset_index()
    entity_counts.columns = ["Entity Type", "Count"]

    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    bar_plot = sns.barplot(data=entity_counts, x="Entity Type", y="Count", palette="viridis")

    for i, row in entity_counts.iterrows():
        bar_plot.text(i, row["Count"], row["Count"], ha='center', va='bottom')

    plt.title("Distribution of Clinical Entity Types")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()
