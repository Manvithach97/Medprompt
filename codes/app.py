import streamlit as st
import pandas as pd
import speech_recognition as sr
import io

from ner import recognize_entities
from filter import filter_entities
from explainer import explain_entities
from llm_extractor import extract_combined
from get_summary import call_summary_llm
from datavis import plot_entity_distribution, save_entity_distribution_chart

from streamlit_mic_recorder import mic_recorder
 # <-- Voice recorder
from pydub import AudioSegment

if "clinical_text" not in st.session_state:
    st.session_state["clinical_text"] = ""

if "voice_text" not in st.session_state:
    st.session_state["voice_text"] = ""


#audio_bytes = None
st.set_page_config(page_title="MedPrompt", layout="wide")

# --- Title ---
st.markdown("""
# MedPrompt  
### Care Coordination and Pathways for Patients using Digital Information
""")

# --- Manual Input ---
with st.expander("🖊️ Manual Input", expanded=True):
    clinical_text = st.text_area("Enter clinical note 📋", height=200, placeholder="Paste raw clinical note here")

# --- Voice Input ---
with st.expander("🎤 Voice Input"):
    st.markdown("Click to record your voice note and convert it to text:")

    audio_bytes = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="🛑 Stop Recording", key='recorder')

    if audio_bytes and isinstance(audio_bytes, dict) and "bytes" in audio_bytes:
        raw_bytes = audio_bytes["bytes"]  # Extract just the audio data
        st.audio(raw_bytes, format="audio/wav")  # ✅ This fixes the binary format error

        try:
            # Convert to PCM WAV using pydub
            audio_segment = AudioSegment.from_file(io.BytesIO(raw_bytes))
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            # Transcribe
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio_data = recognizer.record(source)
                voice_text = recognizer.recognize_google(audio_data)

            st.success("Voice Transcribed Successfully:")
            st.write(voice_text)

            if st.button("Use Voice Input"):
                st.session_state["voice_text"] = voice_text
                st.success("Voice input saved! Now click 'Proceed ➡️' to continue.")


        except sr.UnknownValueError:
            st.error("Could not understand the audio.")
        except sr.RequestError:
            st.error("Speech Recognition service is unavailable.")
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
    else:
        st.info("Please record your voice to proceed.")
        


# --- Main Processing ---
input_text = clinical_text.strip() if clinical_text.strip() else ""
voice_text = st.session_state.get("voice_text", "").strip()

input_text = input_text or voice_text
if st.button("Proceed ➡️") and input_text.strip():
    clinical_text = input_text 
    st.markdown("---")
    st.subheader("🧠 Clinical Entity Recognition")

    with st.spinner("Extracting and filtering entities..."):
        entities = recognize_entities(clinical_text)
        filtered_entities = filter_entities(entities)

    if not filtered_entities:
        st.warning("No relevant clinical entities detected.")
    else:
        df = pd.DataFrame(filtered_entities, columns=["Entity", "Entity Type"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("📘 Clinical Entity Definitions")
        definitions = explain_entities(filtered_entities)
        for definition in definitions:
            if definition:
                st.markdown(f"{definition}")

        st.markdown("---")
        st.subheader("📑 Structured Information Extraction")

        with st.spinner("Calling LLM for structured output..."):
            structured_output = extract_combined(clinical_text)

        if structured_output and isinstance(structured_output, dict):
            concise = structured_output.get("ConciseSummary") or structured_output.get("conciseSummary")
            detailed = structured_output.get("DetailedSummary") or structured_output.get("detailedSummary")

            if concise or detailed:
                with st.expander("🧾 Concise Summary", expanded=True):
                    st.json(concise)
                with st.expander("📄 Detailed Summary", expanded=True):
                    st.json(detailed)

                st.markdown("---")
                st.subheader("🖼️ Summary and Visualization")

                with st.spinner("Generating short summary..."):
                    short_summary = call_summary_llm(clinical_text)

                st.markdown("### 📝 Short Summary")
                st.markdown(f"> {short_summary}")

                plot_entity_distribution(filtered_entities)
                chart_path = save_entity_distribution_chart(filtered_entities)

                # Cache results
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

        # Debug Raw Output
        if isinstance(structured_output, dict) and "RawResponse" in structured_output:
            st.markdown("### 🐞 Raw LLM Output (for debugging)")
            st.text(structured_output["RawResponse"])
