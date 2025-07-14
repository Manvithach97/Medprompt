import streamlit as st
import pandas as pd

from ner import recognize_entities
from filter import filter_entities
from explainer import explain_entities

from llm_extractor import extract_combined
from get_summary import call_summary_llm
from explainer import explain_entities

from datavis import plot_entity_distribution, save_entity_distribution_chart




st.set_page_config(page_title="MedPrompt", layout="wide")

#title
st.markdown("""
# MedPrompt  
### Care Coordination and Pathways for Patients using Digital Information
""")

#input clinical note
clinical_text = st.text_area("Enter clinical note 📋",height=200, placeholder="Paste raw clinical note here")

#main button
if st.button("Proceed ➡️") and clinical_text.strip():
    st.markdown("---")
    st.subheader(" Clinical Entity Recognition")

    with st.spinner("Extracting and filtering entities..."):
        entities = recognize_entities(clinical_text)
        
        entity_count = len(entities)

        filtered_entities = filter_entities(entities)

    if not filtered_entities:
        st.warning("No relevant clinical entities detected.")
    else:
        # Display extracted entities
        df = pd.DataFrame(filtered_entities, columns=["Entity","Entity Type"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader(" Clinical Entity Definitions")
        definitions = explain_entities(filtered_entities)
        for definition in definitions:
            if definition==None:
                continue
            st.markdown(f"{definition}")

        st.markdown("---")
        st.subheader("📑 Structured Information Extraction")

        with st.spinner("Calling LLM for structured output..."):
            structured_output=extract_combined(clinical_text)

        if structured_output and isinstance(structured_output, dict):
            concise = structured_output.get("ConciseSummary") or structured_output.get("conciseSummary")
            detailed = structured_output.get("DetailedSummary") or structured_output.get("detailedSummary")

            if concise is not None or detailed is not None:

                with st.expander("Concise Summary", expanded=True):
                    st.json(concise)

                with st.expander("Detailed Summary", expanded=True):
                    st.json(detailed)

                st.markdown("---")
                st.subheader("Summary and Visualization")

                with st.spinner("Generating short summary..."):
                    short_summary = call_summary_llm(clinical_text)

                st.markdown("### Short Summary")
                st.markdown(f"> {short_summary}")

                # entity distribution chart
                plot_entity_distribution(filtered_entities)
                chart_path = save_entity_distribution_chart(filtered_entities)

                #Save to session_state-avoid recomputation
                st.session_state["entities"] = filtered_entities
                st.session_state["definitions"] = definitions
                st.session_state["concise"] = concise
                st.session_state["detailed"] = detailed
                st.session_state["summary_text"] = short_summary
                st.session_state["chart_path"] = chart_path
            else:
                st.error("Failed to extract structured summary from the model.")
        else:
            st.error("Failed to extract structured output from the LLM.")

        #raw output from LLM for debugging if available
        if isinstance(structured_output, dict) and "RawResponse" in structured_output:
            st.markdown("### Raw LLM Output (for debugging)")
            st.text(structured_output["RawResponse"])
