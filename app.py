import streamlit as st
import time
import google.generativeai as genai
import io

try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` ライブラリがインストールされていません。`pip install openai` を実行してください。")
    OpenAI = None

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    st.error("`streamlit-mic-recorder` ライブラリがインストールされていません。`pip install streamlit-mic_recorder` を実行してください。")
    mic_recorder = None

# --- 0. 環境変数の設定 ---
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Error: GEMINI_API_KEY is not set in Streamlit Secrets.")
    st.stop()

openai_client = None
if OpenAI:
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        openai_client = OpenAI(api_key=openai_api_key)
    except KeyError:
        st.warning("Warning: OPENAI_API_KEY is not set in Streamlit Secrets. Voice input (Whisper) will not be available.")
else:
    openai_client = None

genai.configure(api_key=gemini_api_key)

# --- キャラクター設定とレベル調整機能 ---
def get_system_instruction(level):
    # Base instruction common to all levels, explicitly enforcing English responses
    base_instruction = "You are an English conversation partner who helps users improve their English skills. **Always respond only in English. Do not use Japanese at all.**"

    if level == "Hana":
        return base_instruction + (
            " Your name is Tanaka Hana. You are a girl from Wakaba Junior High School, originally from Wakaba City." # 修正箇所: 名前の順序を変更
            " You have a gentle and meticulous personality, and your friends often consult you when they're in trouble."
            " You've been dedicated to soccer since age 3. Recently, you've been enjoying family camping trips and mastering camp cooking."
            " You're preparing to play in an overseas soccer league after junior high school graduation."
            " Your favorite subject is English, and your hobbies are soccer and baking sweets."
            " You will converse according to the English ability of a Japanese junior high school 1st grader."
            " Focus on basic vocabulary like 'be, have, go, see, eat, school, friend, happy, kind, clean, big, small', targeting a total vocabulary of around 300-1300 words."
            " Speak slowly using very simple words and short sentences (maximum 10 words per sentence)."
            " Ask simple questions to encourage conversation."
            " **Do not point out any grammar or spelling mistakes in the user's input. Accept them as they are and continue the conversation.**"
        )
    elif level == "Mark":
        return base_instruction + (
            " Your name is Mark Davis. You are a boy from Wakaba Junior High School, originally from Seattle, USA."
            " You have a cheerful personality and are a mood-maker in class. You have an older sister who is in high school."
            " You love interacting with people and have been entrusted with looking after the new first-year students in your basketball club."
            " While continuing your beloved basketball, you are diligently studying to become a veterinarian."
            " Your favorite subject is Science, and you are very athletic, placing high in the Wakaba Marathon every year."
            " You will converse according to the English ability of a Japanese junior high school graduate (Eiken Grade 3 equivalent)."
            " Use everyday, emotional, and regional vocabulary such as 'enjoy, plan, decide, describe, delicious, exciting, important, healthy, wonderful, popular', targeting a total vocabulary of around 1250-2100 words."
            " **Only if there are obvious grammar or spelling mistakes in the user's input, gently point them out or suggest a more natural way to phrase it, assisting the user to correct them on their own.**"
            " Incorporate slightly longer sentences and somewhat complex sentence structures, focusing on a natural flow of conversation."
        )
    elif level == "Ms. Brown":
        return base_instruction + (
            " Your name is Ms. Lucy Brown. You are an ALT (Assistant Language Teacher) at Wakaba Junior High School, originally from London, UK."
            " You love reading and own many different books. Recently, you've been reading a lot of Japanese novels."
            " When you were a junior high school student, your dream was to be a novelist, and you often wrote novels based on everyday events."
            " You love houseplants and animals."
            " You will converse according to the English ability of a Japanese English teacher (Eiken Pre-1st Grade, TOEFL PBT 550+, CBT 213+, iBT 80+, TOEIC 730+)."
            " Use professional and abstract vocabulary suitable for university-level studies, specifically targeting words like 'accommodate, acknowledge, eliminate, prohibit, uphold, magnify, acquisition, curriculum, literacy, heritage, ailment, revenue', with a total vocabulary of around 7500-9000 words."
            " If there are grammar or spelling mistakes in the user's input, **gently point them out or suggest more sophisticated expressions, assisting the user to think and correct them on their own.**"
            " However, your role is primarily a facilitator, encouraging the user's critical thinking and expression. Discuss a wide range of topics deeply and in natural English."
        )
    else: # Default case, though unlikely with selectbox
        return base_instruction + " Use natural, everyday English. Engage in friendly conversation and ask open-ended questions."

# --- Streamlit UIの構築 ---
st.set_page_config(layout="wide")
st.title("English Conversation Partner 🗣️")
st.write("Let's practice English together!")

with st.sidebar:
    st.header("Settings")
    
    # 英語レベル選択
    english_level = st.selectbox(
        "Select your English Level:",
        [
            "Hana",
            "Mark",
            "Ms. Brown"
        ],
        index=0,
        key="english_level_selector"
    )

    # 音声入力のON/OFFトグル
    use_audio_input = st.toggle("Enable Voice Input", value=False, key="audio_input_toggle")

    if use_audio_input and (mic_recorder is None or openai_client is None):
        st.warning("音声入力は、`streamlit-mic-recorder` または `openai` ライブラリが不足しているか、OpenAI APIキーが設定されていないため無効です。") # User-facing warning in Japanese is fine
        use_audio_input = False

    st.info("The AI will always respond in English, based on your selected level.")

# --- モデルの初期化 (レベル選択に応じて system_instruction を設定) ---
current_system_instruction = get_system_instruction(english_level)
model = genai.GenerativeModel(
    'gemini-flash-latest',
    system_instruction=current_system_instruction
)

# --- チャット履歴をStreamlitのセッションステートで管理 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial message can vary slightly based on level/character, but a generic one for now.
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your English conversation partner. What would you like to talk about today?"})
    st.session_state.previous_english_level = english_level # Initialize previous_english_level

if st.session_state.get("previous_english_level") != english_level:
    st.session_state.messages = []
    # Dynamic initial message after level change
    initial_message = "Hello! Let's start our conversation. What's on your mind today?"
    if english_level == "Beginner (Junior High School 1st Grade Equivalent)":
        initial_message = "Hi! I'm Tanaka Hana. What would you like to talk about today?" # 修正箇所: ここも名前の順序を変更
    elif english_level == "Intermediate (Junior High School Graduate/Eiken Grade 3 Equivalent)":
        initial_message = "Hey there! I'm Mark. What's up?"
    elif english_level == "Advanced (Japanese English Teacher/Eiken Pre-1st Grade or Higher)":
        initial_message = "Good day! I'm Ms. Brown. How may I assist you today?"

    st.session_state.messages.append({"role": "assistant", "content": f"Okay, switching to the {english_level} level. {initial_message}"})
    st.session_state.previous_english_level = english_level

# --- 既存のチャット履歴を表示 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ユーザーからの入力を受け付ける ---
user_input_prompt = ""
if use_audio_input:
    st.write("Click the mic and speak!")
    audio_bytes = None
    if mic_recorder:
        recorded_audio = mic_recorder(
            start_prompt="🎤 Start recording",
            stop_prompt="⏹️ Stop recording",
            just_once=True,
            use_container_width=True,
            key='user_mic_input'
        )
        if recorded_audio:
            audio_bytes = recorded_audio['bytes']

    if audio_bytes and openai_client:
        with st.spinner("Transcribing audio..."):
            try:
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "audio.wav"
                
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
                user_input_prompt = transcript.text
                st.write(f"You said: {user_input_prompt}")
            except Exception as e:
                st.error(f"Error transcribing audio: {e}")
                user_input_prompt = ""
    
    if not user_input_prompt:
        user_input_prompt = st.chat_input("Start practicing English with me! (Or use mic above)", disabled=bool(audio_bytes))

else:
    user_input_prompt = st.chat_input("Start practicing English with me! (Type here)")

if user_input_prompt:
    st.session_state.messages.append({"role": "user", "content": user_input_prompt})
    with st.chat_message("user"):
        st.markdown(user_input_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        gemini_chat_history = []
        # Exclude initial message when reconstructing chat history for Gemini
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                gemini_chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                if "Okay, switching to the " not in msg["content"]: 
                    gemini_chat_history.append({"role": "model", "parts": [msg["content"]]})


        chat = model.start_chat(history=gemini_chat_history)

        try:
            response = chat.send_message(user_input_prompt, stream=True)

            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"An error occurred with Gemini: {e}. Please try again.")
            st.session_state.messages.append({"role": "assistant", "content": f"An error occurred with Gemini: {e}. Please try again."})