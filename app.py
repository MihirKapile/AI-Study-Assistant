import streamlit as st
import json
import asyncio
import os
from agno.agent import Agent
from agno.models.groq import Groq
from dotenv import load_dotenv
import re
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import tempfile
import io # Added for BytesIO

load_dotenv()

st.set_page_config(layout="wide", page_title="AI Study Assistant", initial_sidebar_state="expanded")

st.markdown("<h1 class='main-header'>🧠 AI Study Assistant 🚀</h1>", unsafe_allow_html=True)
st.write("Organize your study with custom sections, topics, and generate targeted study maps or quizzes!")

if 'sections' not in st.session_state:
    st.session_state.sections = []
if 'study_map_output' not in st.session_state:
    st.session_state.study_map_output = {}
if 'quiz_output' not in st.session_state:
    st.session_state.quiz_output = {}
if 'user_quiz_answers' not in st.session_state:
    st.session_state.user_quiz_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = {}
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = {}
if 'quiz_difficulty_state' not in st.session_state:
    st.session_state.quiz_difficulty_state = {}
if 'current_quiz_question_data' not in st.session_state:
    st.session_state.current_quiz_question_data = {}
if 'quiz_question_feedback' not in st.session_state:
    st.session_state.quiz_question_feedback = {}
if 'quiz_total_correct' not in st.session_state:
    st.session_state.quiz_total_correct = {}
if 'quiz_total_attempted' not in st.session_state:
    st.session_state.quiz_total_attempted = {}
if 'quiz_question_count' not in st.session_state:
    st.session_state.quiz_question_count = {}
if 'active_quiz_section' not in st.session_state:
    st.session_state.active_quiz_section = None
if 'quiz_current_grade' not in st.session_state:
    st.session_state.quiz_current_grade = {}
# NEW: State to lock quiz options
if 'quiz_options_locked' not in st.session_state:
    st.session_state.quiz_options_locked = {}


if 'overall_total_correct' not in st.session_state:
    st.session_state.overall_total_correct = 0
if 'overall_total_attempted' not in st.session_state:
    st.session_state.overall_total_attempted = 0
if 'overall_grade' not in st.session_state:
    st.session_state.overall_grade = "N/A"

if 'voice_input_subject' not in st.session_state:
    st.session_state.voice_input_subject = ""

if 'tts_audio_files' not in st.session_state:
    st.session_state.tts_audio_files = {}

MAX_QUIZ_QUESTIONS = 10

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY environment variable not set. Please set it to use Groq API.")
    st.stop()

@st.cache_resource
def get_study_map_agent():
    return Agent(
        model=Groq(id="llama3-8b-8192"),
        name="Study Map Agent",
        role="An expert educator focused on breaking down complex topics into digestible study plans.",
        instructions=[
            "Create a structured and comprehensive study map.",
            "Include Key Concepts (main ideas).",
            "Include Sub-topics for each key concept.",
            "Provide a brief, 1-2 sentence explanation for each sub-topic.",
            "List Important terms or vocabulary related to the sub-topics, in parentheses after the explanation.",
            "Format the output clearly using Markdown. Use headings for Key Concepts, bullet points for sub-topics, and bold for important terms.",
            """Example structure:
            ## Key Concept 1: [Name]
            * **Sub-topic 1.1**: Brief explanation. (Terms: Term1, Term2)
            * **Sub-topic 1.2**: Brief explanation. (Terms: Term3)

            ## Key Concept 2: [Name]
            * **Sub-topic 2.1**: Brief explanation. (Terms: Term4)
            """
        ],
        markdown=True
    )

@st.cache_resource
def get_quiz_generation_agent():
    return Agent(
        model=Groq(id="llama3-8b-8192"),
        name="Quiz Generation Agent",
        role="A specialized AI for creating accurate and engaging educational quizzes covering multiple related topics.",
        instructions=[
            "Generate a SINGLE multiple-choice question.",
            "The question should have four answer options (A, B, C, D).",
            "Provide the correct answer clearly marked, and a brief explanation for the correct answer.",
            "Format the question clearly using Markdown.",
            "Based on the difficulty hint, adjust the complexity of the question and options.",
            """Example:
            ### Question 1:
            What is the capital of France?
            A) Berlin
            B) Madrid
            C) Paris
            D) Rome
            Correct Answer: C) Paris
            Explanation: Paris is the capital and most populous city of France.
            """
        ],
        markdown=True
    )

@st.cache_resource
def get_curriculum_agent():
    curriculum_schema_instruction = """
    The output MUST be a JSON object ONLY, with the following structure. Do NOT include any other text or markdown.
    {
      "sections": [
        {
          "name": "Section Name",
          "topics": [
            "Topic 1",
            "Topic 2"
          ]
        }
      ]
    }
    """
    return Agent(
        model=Groq(id="llama3-8b-8192"),
        name="Curriculum Agent",
        role="An expert curriculum designer whose task is to break down a broad subject into logical sections and relevant topics within each section. The output must be a pure JSON object.",
        instructions=[
            "Focus on core concepts and foundational knowledge for the given subject.",
            "Provide at least 5 sections, with 4-6 relevant topics per section.",
            "Ensure the topics within each section are cohesive.",
            "Your entire response MUST be a valid JSON object. Do NOT include any conversational text, introductory phrases, or markdown outside the JSON structure.",
            curriculum_schema_instruction
        ],
        markdown=False
    )

study_map_agent = get_study_map_agent()
quiz_generation_agent = get_quiz_generation_agent()
curriculum_agent = get_curriculum_agent()

def parse_single_quiz_question_markdown(markdown_text):
    match = re.search(
        r'^(?:### Question \d+:)?\s*(.*?)\n'
        r'([A-D]\).*?)\n'
        r'([A-D]\).*?)\n'
        r'([A-D]\).*?)\n'
        r'([A-D]\).*?)\n'
        r'Correct Answer: ([A-D]\) .*?)\n'
        r'Explanation: (.*)$',
        markdown_text, re.DOTALL | re.MULTILINE
    )

    if match:
        question_text = match.group(1).strip()
        options = [
            match.group(2).strip(),
            match.group(3).strip(),
            match.group(4).strip(),
            match.group(5).strip()
        ]
        correct_answer_full = match.group(6).strip()
        explanation = match.group(7).strip()

        correct_answer_letter_match = re.match(r'([A-D])\)', correct_answer_full)
        correct_answer_letter = correct_answer_letter_match.group(1) if correct_answer_letter_match else None

        return {
            "question": question_text,
            "options": options,
            "correct_answer_full": correct_answer_full,
            "correct_answer_letter": correct_answer_letter,
            "explanation": explanation
        }
    return None

def generate_adaptive_quiz_question(section_name, all_topics_in_section, difficulty_hint="normal"):
    current_topic_index = st.session_state.quiz_difficulty_state[section_name].get('current_topic_index', 0)
    
    if not all_topics_in_section:
        st.warning("No topics available in this section for quiz generation.")
        return None

    if current_topic_index < 0:
        current_topic_index = 0
    if current_topic_index >= len(all_topics_in_section):
        current_topic_index = len(all_topics_in_section) - 1

    current_topic = all_topics_in_section[current_topic_index]
    
    st.session_state.quiz_difficulty_state[section_name]['current_topic_index'] = current_topic_index
    st.session_state.quiz_difficulty_state[section_name]['current_topic'] = current_topic

    prompt_instructions = ""
    if difficulty_hint == "easier":
        prev_topic_index = max(0, current_topic_index - 1)
        target_topic = all_topics_in_section[prev_topic_index]
        
        if prev_topic_index != current_topic_index:
            st.session_state.quiz_difficulty_state[section_name]['current_topic_index'] = prev_topic_index
            st.session_state.quiz_difficulty_state[section_name]['current_topic'] = target_topic
            prompt_instructions = f"Generate an EASIER question about the PREVIOUS or CURRENT related topic: '{target_topic}'. Focus on foundational concepts or simpler aspects of this topic."
        else:
            prompt_instructions = f"Generate an EASIER question about '{current_topic}'. Focus on foundational concepts or simpler aspects of this topic."
            
    elif difficulty_hint == "harder":
        next_topic_index = min(current_topic_index + 1, len(all_topics_in_section) - 1)
        target_topic = all_topics_in_section[next_topic_index]
        
        if next_topic_index != current_topic_index:
            st.session_state.quiz_difficulty_state[section_name]['current_topic_index'] = next_topic_index
            st.session_state.quiz_difficulty_state[section_name]['current_topic'] = target_topic
            prompt_instructions = f"Generate a question about the NEXT related topic: '{target_topic}'. Focus on core concepts."
        else:
            prompt_instructions = f"Generate a HARDER question about '{current_topic}'. Introduce more complex or nuanced aspects."
    else:
        prompt_instructions = f"Generate a question about the topic: '{current_topic}'. Focus on a balanced difficulty."

    st.toast(f"Generating question on '{st.session_state.quiz_difficulty_state[section_name]['current_topic']}' ({difficulty_hint})...")
    
    with st.spinner("Generating question..."):
        try:
            quiz_response = quiz_generation_agent.run(prompt_instructions)
            return quiz_response.content
        except Exception as e:
            st.error(f"Error generating adaptive quiz question: {e}")
            return None

def calculate_overall_grade():
    if st.session_state.overall_total_attempted > 0:
        overall_grade_val = (st.session_state.overall_total_correct / st.session_state.overall_total_attempted) * 100
        st.session_state.overall_grade = f"{overall_grade_val:.1f}%"
    else:
        st.session_state.overall_grade = "N/A"

def check_answer_and_adjust_difficulty(section_name, user_selected_option, current_q_data):
    is_correct = False
    feedback_message = ""
    
    if user_selected_option is not None:
        if current_q_data:
            user_choice_letter = user_selected_option[0] if user_selected_option else None

            if user_choice_letter == current_q_data['correct_answer_letter']:
                is_correct = True
                feedback_message = f"**Correct!** {current_q_data['explanation']}"
                st.session_state.quiz_total_correct[section_name] += 1
                st.session_state.overall_total_correct += 1
            else:
                feedback_message = f"**Incorrect.** The correct answer was **{current_q_data['correct_answer_full']}**. {current_q_data['explanation']}"
            st.session_state.quiz_total_attempted[section_name] += 1
            st.session_state.overall_total_attempted += 1
        else:
            feedback_message = "Error: Question data missing for evaluation."
    else:
        feedback_message = "Please select an answer before checking."

    st.session_state.quiz_question_feedback[section_name] = feedback_message
    st.session_state.quiz_options_locked[section_name] = True # Lock options after checking

    if is_correct:
        st.session_state.quiz_difficulty_state[section_name]['difficulty_hint'] = "harder"
    else:
        st.session_state.quiz_difficulty_state[section_name]['difficulty_hint'] = "easier"

    if st.session_state.quiz_total_attempted[section_name] > 0:
        grade = (st.session_state.quiz_total_correct[section_name] / st.session_state.quiz_total_attempted[section_name]) * 100
        st.session_state.quiz_current_grade[section_name] = f"{grade:.1f}%"
    else:
        st.session_state.quiz_current_grade[section_name] = "N/A"
    
    calculate_overall_grade()

    st.rerun()

def text_to_speech_and_play(text, key_id):
    with st.spinner("Generating audio..."):
        try:
            tts = gTTS(text=text, lang='en')
            
            audio_bytes_io = io.BytesIO()
            tts.write_to_fp(audio_bytes_io)
            audio_bytes_io.seek(0)

            st.audio(audio_bytes_io.getvalue(), format="audio/mp3", loop=False, autoplay=True)
            st.success("Audio generated and playing!")
        except Exception as e:
            st.error(f"Error generating audio: {e}")


st.markdown("<h2 style='text-align: center;'>Generate Full Study Curriculum</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    main_study_subject = st.text_input(
        "Enter a main study subject (e.g., 'Java Programming', 'World History'):", 
        key="main_subject_input",
        value=st.session_state.voice_input_subject 
    )
with col2:
    st.write("Or speak it:")
    voice_transcript = speech_to_text(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True,
        use_container_width=True,
        key='voice_subject_stt'
    )
    if voice_transcript:
        st.session_state.voice_input_subject = voice_transcript
        st.rerun()

if st.button("Generate Curriculum", key="generate_curriculum_btn"):
    if main_study_subject:
        with st.spinner(f"Generating curriculum for '{main_study_subject}'... This might take a moment."):
            try:
                curriculum_raw_content = curriculum_agent.run(f"Generate a curriculum for the subject: '{main_study_subject}'").content
                curriculum_data = json.loads(curriculum_raw_content)

                if isinstance(curriculum_data, dict) and "sections" in curriculum_data:
                    st.session_state.sections = curriculum_data["sections"]
                    st.success(f"Curriculum generated for '{main_study_subject}'!")
                    st.toast("Curriculum generated successfully!")
                else:
                    st.error("Failed to parse curriculum from agent. Response was not in expected JSON format or missing 'sections' key.")
                    st.json(curriculum_data)
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON from Curriculum Agent. The AI might not have returned pure JSON. Details: {e}")
                st.text(f"Raw content received: \n{curriculum_raw_content}")
            except Exception as e:
                st.error(f"An unexpected error occurred while generating curriculum with Curriculum Agent: {e}")
    else:
        st.warning("Please enter a main study subject to generate a curriculum.")

st.markdown("---")

with st.sidebar:
    st.header("Overall Performance")
    overall_grade_display = st.session_state.overall_grade
    st.markdown(f"**Overall Grade:** <span style='font-size: 2em; font-weight: bold;'>{overall_grade_display}</span>", unsafe_allow_html=True)
    st.markdown(f"**Total Correct:** {st.session_state.overall_total_correct}")
    st.markdown(f"**Total Attempted:** {st.session_state.overall_total_attempted}")
    st.info("This grade reflects your performance across all adaptive quizzes.")

st.sidebar.markdown("---")
st.sidebar.header("Manage Study Content (Manual)")
new_section_name = st.sidebar.text_input("New Section Name", key="new_section_input")
if st.sidebar.button("Add New Section", key="add_section_btn"):
    if new_section_name and new_section_name not in [s['name'] for s in st.session_state.sections]:
        st.session_state.sections.append({"name": new_section_name, "topics": []})
        st.sidebar.success(f"Section '{new_section_name}' added!")
    elif new_section_name:
        st.sidebar.warning("Section with this name already exists.")
    else:
        st.sidebar.warning("Please enter a section name.")

st.sidebar.markdown("---")
st.sidebar.header("Add Topics to Sections (Manual)")
if st.session_state.sections:
    section_names = [s['name'] for s in st.session_state.sections]
    selected_section_for_topic = st.sidebar.selectbox("Select a Section", section_names, key="select_section_for_topic")
    new_topic_name = st.sidebar.text_input("New Topic Name", key="new_topic_input")

    if st.sidebar.button("Add Topic to Selected Section", key="add_topic_btn"):
        if new_topic_name and selected_section_for_topic:
            for section in st.session_state.sections:
                if section['name'] == selected_section_for_topic:
                    if new_topic_name not in section['topics']:
                        section['topics'].append(new_topic_name)
                        st.sidebar.success(f"Topic '{new_topic_name}' added to '{selected_section_for_topic}'.")
                    else:
                        st.sidebar.warning("Topic already exists in this section.")
                    break
        else:
            st.sidebar.warning("Please enter a topic name and select a section.")
else:
    st.sidebar.info("Add a section first to add topics (or generate a curriculum above).")

st.sidebar.markdown("---")

st.markdown("<h2>Your Study Plan</h2>", unsafe_allow_html=True)

if not st.session_state.sections:
    st.info("Start by generating a curriculum above, or manually adding a new section from the sidebar!")

for i, section_data in enumerate(st.session_state.sections):
    section_name = section_data['name']
    topics_in_section = section_data['topics']
    section_id_key = section_name.replace(" ", "_").replace(":", "").replace("/", "").replace(".", "").replace("-", "_")

    with st.container(border=True):
        st.markdown(f"<h3>Section: {section_name}</h3>", unsafe_allow_html=True)

        if section_name not in st.session_state.quiz_difficulty_state:
            st.session_state.quiz_difficulty_state[section_name] = {
                'current_topic_index': 0,
                'difficulty_hint': "normal",
                'current_topic': topics_in_section[0] if topics_in_section else "general knowledge"
            }
        
        if section_name not in st.session_state.quiz_total_correct:
            st.session_state.quiz_total_correct[section_name] = 0
        if section_name not in st.session_state.quiz_total_attempted:
            st.session_state.quiz_total_attempted[section_name] = 0
        if section_name not in st.session_state.quiz_question_count:
            st.session_state.quiz_question_count[section_name] = 0
        if section_name not in st.session_state.quiz_current_grade:
            st.session_state.quiz_current_grade[section_name] = "N/A"
        # Initialize quiz_options_locked for this section
        if section_name not in st.session_state.quiz_options_locked:
            st.session_state.quiz_options_locked[section_name] = False


        if st.button(f"Start Adaptive Quiz for '{section_name}'", key=f"start_quiz_btn_{section_id_key}"):
            if topics_in_section:
                st.session_state[f'quiz_loading_{section_id_key}'] = True
                st.session_state.quiz_submitted[section_name] = False
                st.session_state.quiz_total_correct[section_name] = 0
                st.session_state.quiz_total_attempted[section_name] = 0
                st.session_state.quiz_question_feedback[section_name] = ""
                st.session_state.quiz_difficulty_state[section_name]['current_topic_index'] = 0
                st.session_state.quiz_difficulty_state[section_name]['difficulty_hint'] = "normal"
                st.session_state.quiz_question_count[section_name] = 0
                st.session_state.quiz_current_grade[section_name] = "N/A"
                st.session_state.active_quiz_section = section_name
                st.session_state.quiz_options_locked[section_name] = False # Unlock options for new quiz

                if topics_in_section:
                    new_question_markdown = generate_adaptive_quiz_question(
                        section_name, 
                        topics_in_section, 
                        st.session_state.quiz_difficulty_state[section_name]['difficulty_hint']
                    )
                    st.session_state.quiz_output[section_name] = new_question_markdown
                    st.session_state.current_quiz_question_data[section_name] = parse_single_quiz_question_markdown(new_question_markdown)
                    if new_question_markdown:
                        st.session_state.quiz_question_count[section_name] += 1
                else:
                    st.warning("Cannot start quiz: No topics found in this section.")
                    st.session_state.quiz_output.pop(section_name, None)
                    st.session_state.current_quiz_question_data.pop(section_name, None)
            else:
                st.warning(f"Please add some topics to section '{section_name}' before starting an adaptive quiz.")
            st.session_state[f'quiz_loading_{section_id_key}'] = False
            st.rerun()

        st.markdown("<h4>Topics:</h4>", unsafe_allow_html=True)
        if not topics_in_section:
            st.info("No topics added to this section yet.")

        for j, topic_name in enumerate(topics_in_section):
            topic_id_key = f"{section_id_key}_{topic_name.replace(' ', '_').replace(':', '').replace('/', '').replace('.', '').replace('-', '_')}_{j}"

            with st.expander(f"Topic: {topic_name}", expanded=False):
                col_study_map_btn, col_tts_btn = st.columns([0.7, 0.3])
                
                with col_study_map_btn:
                    if st.button(f"Generate Study Map for '{topic_name}'", key=f"study_map_btn_{topic_id_key}"):
                        st.session_state[f'study_map_loading_{topic_id_key}'] = True
                        with st.spinner(f"Generating study map for topic: {topic_name}..."):
                            try:
                                study_map_response = study_map_agent.run(f"Generate a study map for the topic: '{topic_name}'")
                                st.session_state.study_map_output[topic_id_key] = study_map_response.content
                            except Exception as e:
                                st.error(f"Error generating study map for topic '{topic_name}': {e}")
                            st.session_state[f'study_map_loading_{topic_id_key}'] = False

                if topic_id_key in st.session_state.study_map_output:
                    with col_tts_btn:
                        if st.button("🔊 Read Aloud", key=f"tts_btn_{topic_id_key}"):
                            tts_text = st.session_state.study_map_output[topic_id_key]
                            # Clean up markdown for better TTS pronunciation
                            tts_text = re.sub(r'##\s*', '', tts_text)
                            tts_text = re.sub(r'\*\*(.*?)\*\*', r'\1', tts_text)
                            tts_text = re.sub(r'\*', '', tts_text)
                            tts_text = re.sub(r'\(Terms:.*?\)', '', tts_text)

                            text_to_speech_and_play(tts_text, topic_id_key)

                    st.markdown("---")
                    st.markdown(f"**Study Map for {topic_name}:**")
                    st.markdown(st.session_state.study_map_output[topic_id_key], unsafe_allow_html=True)

        if st.session_state.active_quiz_section == section_name and not st.session_state.quiz_submitted[section_name]:
            current_q_data = st.session_state.current_quiz_question_data.get(section_name)

            if current_q_data:
                st.markdown("---")
                st.markdown(f"**Question {st.session_state.quiz_question_count[section_name]} of {MAX_QUIZ_QUESTIONS}**")
                st.markdown(f"**Current Topic:** {st.session_state.quiz_difficulty_state[section_name]['current_topic']}")

                st.markdown(f"**Question:** {current_q_data['question']}")

                # Only allow selection if options are not locked
                user_selected_option = st.radio(
                    "Select your answer:",
                    options=current_q_data['options'],
                    key=f"adaptive_quiz_q_radio_{section_id_key}",
                    index=None,
                    disabled=st.session_state.quiz_options_locked[section_name] # Disable when locked
                )
                
                # Only show check answer button if options are not locked
                if not st.session_state.quiz_options_locked[section_name]:
                    if st.button("Check Answer", key=f"check_answer_btn_{section_id_key}"):
                        check_answer_and_adjust_difficulty(section_name, user_selected_option, current_q_data)

                if st.session_state.quiz_question_feedback.get(section_name):
                    st.markdown(st.session_state.quiz_question_feedback[section_name], unsafe_allow_html=True)
                    
                    if st.session_state.quiz_question_count[section_name] < MAX_QUIZ_QUESTIONS:
                        next_button_label = "Next Question"
                    else:
                        next_button_label = "Finish Quiz"

                    # Only show next/finish button after an answer has been checked (options are locked)
                    if st.session_state.quiz_options_locked[section_name]:
                        if st.button(next_button_label, key=f"next_adaptive_q_btn_{section_id_key}"):
                            st.session_state.quiz_question_feedback[section_name] = ""
                            st.session_state.quiz_options_locked[section_name] = False # Unlock for next question
                            
                            if st.session_state.quiz_question_count[section_name] < MAX_QUIZ_QUESTIONS:
                                new_question_markdown = generate_adaptive_quiz_question(
                                    section_name,
                                    topics_in_section,
                                    st.session_state.quiz_difficulty_state[section_name]['difficulty_hint']
                                )
                                st.session_state.quiz_output[section_name] = new_question_markdown
                                st.session_state.current_quiz_question_data[section_name] = parse_single_quiz_question_markdown(new_question_markdown)
                                if new_question_markdown:
                                    st.session_state.quiz_question_count[section_name] += 1
                                calculate_overall_grade()
                                st.rerun()
                            else:
                                st.session_state.quiz_submitted[section_name] = True
                                st.session_state.active_quiz_section = None
                                calculate_overall_grade()
                                st.rerun()
            else:
                st.info("No current question. Click 'Start Adaptive Quiz' above to begin.")
        elif st.session_state.quiz_submitted.get(section_name):
            st.success(f"Adaptive Quiz Session for '{section_name}' Completed!")
            final_grade = st.session_state.quiz_current_grade.get(section_name, "N/A")
            st.markdown(f"**Final Score for this quiz:** {st.session_state.quiz_total_correct[section_name]} / {st.session_state.quiz_total_attempted[section_name]}")
            st.markdown(f"**Final Grade for this quiz:** <span style='font-size: 1.5em; font-weight: bold;'>{final_grade}</span>", unsafe_allow_html=True)
            st.info("You can restart the quiz to try again with new questions.")

            if st.button(f"Restart Adaptive Quiz for '{section_name}'", key=f"restart_adaptive_quiz_{section_id_key}"):
                st.session_state.quiz_output.pop(section_name, None)
                st.session_state.current_quiz_question_data.pop(section_name, None)
                st.session_state.quiz_question_feedback.pop(section_name, None)
                st.session_state.quiz_submitted[section_name] = False
                st.session_state.quiz_total_correct[section_name] = 0
                st.session_state.quiz_total_attempted[section_name] = 0
                st.session_state.quiz_question_count[section_name] = 0
                st.session_state.quiz_difficulty_state[section_name]['current_topic_index'] = 0
                st.session_state.quiz_difficulty_state[section_name]['difficulty_hint'] = "normal"
                st.session_state.quiz_current_grade[section_name] = "N/A"
                st.session_state.active_quiz_section = None
                st.session_state.quiz_options_locked[section_name] = False
                calculate_overall_grade()
                st.rerun()