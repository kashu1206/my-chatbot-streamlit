import base64
import streamlit as st
import time
import google.generativeai as genai
import io
import os
import json

# GCP Speech-to-Text and Text-to-Speech clients
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from google.oauth2 import service_account

# VAD (Voice Activity Detection) libraries
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    st.error("`streamlit-mic_recorder` ライブラリがインストールされていません。`pip install streamlit-mic-recorder` を実行してください。")
    mic_recorder = None

# --- 0. 環境変数の設定とクライアントの初期化 ---

# Gemini API Key
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
except KeyError:
    st.error("Error: GEMINI_API_KEY is not set in Streamlit Secrets.")
    st.stop() # Gemini APIキーがなければアプリを停止

# GCPクライアントの初期化変数を定義
_tts_client = None
_stt_client = None
_can_use_gcp_voice = False
_decoded_gcp_credentials_json_string = None # 認証情報を格納する変数

try:
    # SecretsからGCP認証情報を読み込む優先順位を設定
    # 1. Base64エンコードされた認証情報
    if "GCP_CREDENTIALS_BASE64" in st.secrets:
        encoded_credentials = st.secrets["GCP_CREDENTIALS_BASE64"]
        _decoded_gcp_credentials_json_string = base64.b64decode(encoded_credentials.encode("utf-8")).decode("utf-8")

    # 2. 直接JSON文字列として設定された認証情報
    elif "GCP_CREDENTIALS" in st.secrets:
        raw_credentials = st.secrets["GCP_CREDENTIALS"]
        if isinstance(raw_credentials, dict):
            _decoded_gcp_credentials_json_string = json.dumps(raw_credentials)
        else:
            _decoded_gcp_credentials_json_string = raw_credentials
    
    # 3. 以前の `GCP_SERVICE_ACCOUNT_KEY` との互換性のため
    elif "GCP_SERVICE_ACCOUNT_KEY" in st.secrets:
        gcp_service_account_key_json = st.secrets.get("GCP_SERVICE_ACCOUNT_KEY")
        if gcp_service_account_key_json:
            service_account_info = json.loads(gcp_service_account_key_json)
            _decoded_gcp_credentials_json_string = json.dumps(service_account_info)
            st.sidebar.success("GCP Credentials loaded successfully from GCP_SERVICE_ACCOUNT_KEY (legacy).")

    # 認証情報がSecretsに見つからなかった場合、デフォルト認証を試みる
    else:
        st.sidebar.warning("Warning: GCP credentials (GCP_CREDENTIALS_BASE64, GCP_CREDENTIALS, or GCP_SERVICE_ACCOUNT_KEY) not found in Streamlit Secrets. Attempting default credentials.")
        try:
            _tts_client = texttospeech.TextToSpeechClient()
            _stt_client = speech.SpeechClient()
            _can_use_gcp_voice = True
            st.sidebar.info("GCP clients initialized with default credentials.")
        except Exception as e:
            st.sidebar.error(f"Failed to initialize GCP clients with default credentials: {e}")
            _can_use_gcp_voice = False
        
    # 読み込んだ認証情報文字列が存在する場合のみ、GCPクライアントを初期化
    if _decoded_gcp_credentials_json_string:
        _gcp_credentials_info = json.loads(_decoded_gcp_credentials_json_string)
        credentials = service_account.Credentials.from_service_account_info(_gcp_credentials_info)
        
        # GCPクライアントを初期化（グローバル変数 _stt_client と _tts_client を使用）
        _stt_client = speech.SpeechClient(credentials=credentials)
        _tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        _can_use_gcp_voice = True
        st.sidebar.info("GCP Speech-to-Text and Text-to-Speech clients initialized from secrets.")

except json.JSONDecodeError as e:
    st.sidebar.error(f"Error decoding GCP credentials JSON: {e}. Please check your Secret format.")
    _can_use_gcp_voice = False
except Exception as e:
    st.sidebar.error(f"Critical error during GCP client setup: {e}")
    _can_use_gcp_voice = False

if not _can_use_gcp_voice:
    st.warning("Voice input/output will not be available due to GCP client initialization failure.")

# --- 音声処理 (無音検出・トリミング) の設定 ---
SAMPLE_RATE = 16000  # Streamlit mic recorder は通常16kHzで録音される (GCP Speech-to-Textの推奨)

# --- 音声からテキストへ (Speech-to-Text) ---
def transcribe_audio_gcp(audio_bytes):
    # クライアントが初期化されているか再確認
    if _stt_client is None:
        st.error("Speech-to-Text client is not initialized. Cannot transcribe audio.")
        return ""

    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        if audio_segment.frame_rate != SAMPLE_RATE or audio_segment.channels != 1:
            audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
        
        nonsilent_chunks = detect_nonsilent(audio_segment, 
                                            min_silence_len=500,
                                            silence_thresh=-35)

        if not nonsilent_chunks:
            st.info("No substantial speech detected after trimming.")
            return ""

        trimmed_audio = AudioSegment.empty()
        for start_ms, end_ms in nonsilent_chunks:
            trimmed_audio += audio_segment[start_ms:end_ms]

        # st.info(f"Original audio duration: {len(audio_segment)/1000:.2f}s, Trimmed audio duration: {len(trimmed_audio)/1000:.2f}s")

        # pydub.AudioSegment のサンプル幅を16-bit (2バイト) に設定
        trimmed_audio = trimmed_audio.set_sample_width(2)

        # ★修正: export() の書き方★
        output_buffer = io.BytesIO()
        trimmed_audio.export(output_buffer, format="wav")
        trimmed_audio_bytes = output_buffer.getvalue()

        audio = speech.RecognitionAudio(content=trimmed_audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        response = _stt_client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        return transcript
    except Exception as e:
        st.error(f"Error transcribing audio with Google Cloud Speech-to-Text API: {e}")
        return ""

# --- テキストから音声へ (Text-to-Speech) ---
def synthesize_text_gcp(text):
    # クライアントが初期化されているか再確認
    if _tts_client is None:
        st.error("Text-to-Speech client is not initialized. Cannot synthesize speech.")
        return None

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Standard-F",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.M4A,
            speaking_rate=1.0,
        )

        response = _tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        st.error(f"Error synthesizing speech with Google Cloud Text-to-Speech API: {e}")
        return None

# --- Geminiモデルの初期化 ---
@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel('gemini-flash-latest')
model = get_gemini_model()

# --- キャラクター設定とレベル調整機能 ---
def get_system_instruction(level):
    # Base instruction common to all levels, explicitly enforcing English responses
    base_instruction = (
    "You are an English conversation partner who helps users improve their English skills. "
    "You are also an experienced English teacher with extensive experience guiding native Japanese speakers in learning English as a foreign language. "
    "Please keep in mind that the user is a native Japanese speaker throughout your interactions. "
    "**Always respond only in English. Do not use Japanese at all.**"
)

    if level == "Hana":
        return base_instruction + (
            " Your name is Tanaka Hana. You are a girl from Wakaba Junior High School, originally from Wakaba City."
            " You have a gentle and meticulous personality, and your friends often consult you when they're in trouble."
            " You've been dedicated to soccer since age 3. Recently, you've been enjoying family camping trips and mastering camp cooking."
            " You're preparing to play in an overseas soccer league after junior high school graduation."
            " Your favorite subject is English, and your hobbies are soccer and baking sweets."
            " You will converse according to the English ability of a Japanese junior high school 1st grader."
            " Focus on basic vocabulary like 'be, have, go, see, eat, school, friend, happy, kind, clean, big, small', targeting a total vocabulary of around 300-1300 words."
            " Speak slowly using very simple words and short sentences (maximum 10 words per sentence)."
            " Ask simple questions to encourage conversation."
            " Keep your responses concise and conversational, ideally around 50 words. Only expand slightly if you need to clarify something briefly."
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
            " Prioritize concise and conversational responses, generally aiming for about 100 words. However, feel free to expand and provide more detail when explaining a concept, sharing an interesting perspective, or offering helpful suggestions related to grammar or vocabulary."
            " **Only if there are obvious grammar or spelling mistakes in the user's input, gently point them out or suggest a more natural way to phrase it, assisting the user to correct them on their own.**"
            " Incorporate slightly longer sentences and somewhat complex sentence structures, focusing on a natural flow of conversation."
        )
    elif level == "Ms. Brown":
        return base_instruction + (
            " Your name is Ms. Lucy Brown. You are an ALT (Assistant Language Teacher) at Wakaba Junior High School, originally from London, UK."
            " You love reading and own many different books. Recently, you've been reading a lot of Japanese novels."
            " When you were a junior high school student, your dream was to be a novelist, and you often wrote novels based on everyday events."
            " You love houseplants and animals."
            " You will converse in a sophisticated and natural English style, appropriate for an English teacher, but always keeping in mind that your user is a Japanese junior high school student." # ここを修正
            " Your responses should be clear, engaging, and aim to gently expand their vocabulary and grammatical understanding without being overwhelming." # ここを修正
            " While you may introduce new, slightly more advanced words or expressions, ensure they are understandable through context or by providing simple explanations if necessary." # ここを追加
            " Avoid overly academic, abstract, or highly specialized vocabulary that would be far beyond a typical junior high school student's comprehension without significant explanation." # ここを追加
            " While your default should be a natural, conversational length to foster dynamic exchange, you are encouraged to expand your responses, typically up to around 200 words, when providing detailed explanations of grammar or vocabulary, offering deeper insights, or giving comprehensive feedback to enhance the user's learning."
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

    # 音声入力/出力のON/OFFトグル (GCPクライアントが初期化できた場合のみ有効)
    use_audio_io = st.toggle("音声入出力", value=False, key="audio_io_toggle", disabled=not _can_use_gcp_voice)

    if use_audio_io and (mic_recorder is None or not _can_use_gcp_voice): # mic_recorderの利用可能性とGCP音声利用可能性の両方を確認
        st.warning("音声入出力は、`streamlit-mic-recorder` ライブラリが不足しているか、GCP認証情報が正しく設定されていないため無効です。")

# --- モデルの初期化 (レベル選択に応じて system_instruction を設定) ---
current_system_instruction = get_system_instruction(english_level)
model = genai.GenerativeModel(
    'gemini-flash-latest',
    system_instruction=current_system_instruction
)

# --- チャット履歴をStreamlitのセッションステートで管理 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.previous_english_level = english_level # Initialize previous_english_level
    # 初回メッセージを生成
    initial_message = ""
    if english_level == "Hana":
        initial_message = "Hi! I'm Tanaka Hana. What would you like to talk about today?"
    elif english_level == "Mark":
        initial_message = "Hey there! I'm Mark. What's up?"
    elif english_level == "Ms. Brown":
        initial_message = "Good day! I'm Ms. Brown. How may I assist you today?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

    # 初回メッセージの音声再生
    if use_audio_io and _can_use_gcp_voice and initial_message:
        audio_output = synthesize_text_gcp(initial_message)
        if audio_output:
            st.audio(audio_output, format="audio/mp4", autoplay=True)


if st.session_state.get("previous_english_level") != english_level:
    st.session_state.messages = []
    # Dynamic initial message after level change
    initial_message = "Hello! Let's start our conversation. What's on your mind today?"
    if english_level == "Hana":
        initial_message = "Hi! I'm Tanaka Hana. What would you like to talk about today?"
    elif english_level == "Mark":
        initial_message = "Hey there! I'm Mark. What's up?"
    elif english_level == "Ms. Brown":
        initial_message = "Good day! I'm Ms. Brown. How may I assist you today?"

    system_change_message = f"Okay, switching to the {english_level} . {initial_message}"
    st.session_state.messages.append({"role": "assistant", "content": system_change_message})
    st.session_state.previous_english_level = english_level

    # レベル変更時のメッセージの音声再生
    if use_audio_io and _can_use_gcp_voice and system_change_message:
        audio_output = synthesize_text_gcp(system_change_message)
        if audio_output:
            st.audio(audio_output, format="audio/mp4", autoplay=True)

# --- 既存のチャット履歴を表示 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ユーザーからの入力を受け付ける ---
user_input_from_mic = ""   # マイクからの入力結果を保持する変数
user_input_from_text = ""  # テキスト入力フォームからの結果を保持する変数
final_user_input_prompt = "" # 最終的にGeminiに送るプロンプト

if use_audio_io:
    st.write("Click the mic and speak!")
    audio_bytes = None
    if mic_recorder: # mic_recorder が利用可能かチェック
        recorded_audio = mic_recorder(
            start_prompt="🎤 録音開始",
            stop_prompt="⏹️ 録音停止",
            just_once=True,
            use_container_width=True,
            key='user_mic_input'
        )
        if recorded_audio:
            audio_bytes = recorded_audio['bytes']

    if audio_bytes and _can_use_gcp_voice:
        with st.spinner("Processing audio and transcribing..."):
            user_input_from_mic = transcribe_audio_gcp(audio_bytes)
            # if user_input_from_mic:
                # st.write(f"You said: {user_input_from_mic}")
            else:
                st.warning("Could not transcribe audio. Please try speaking clearer, or use text input below.")
    elif audio_bytes and not _can_use_gcp_voice: # 音声データがあるがGCPが使えない場合
        st.warning("GCP voice services are not enabled. Cannot transcribe recorded audio.")

    # マイクからの入力があったかどうかにかかわらず、テキスト入力フォームは常に表示・有効
    # ここでの disabled は常に False となる
    user_input_from_text = st.chat_input("Start practicing English with me! (Type here)")

    # 最終的なユーザー入力を決定：マイクからの入力があればそれを優先、なければテキスト入力を使う
    if user_input_from_mic:
        final_user_input_prompt = user_input_from_mic
    elif user_input_from_text:
        final_user_input_prompt = user_input_from_text

else: # use_audio_io が False の場合 (音声入力が無効な場合)
    final_user_input_prompt = st.chat_input("Start practicing English with me! (Type here)")


if final_user_input_prompt: # ★ここを final_user_input_prompt に変更★
    st.session_state.messages.append({"role": "user", "content": final_user_input_prompt}) # ★ここを final_user_input_prompt に変更★
    with st.chat_message("user"):
        st.markdown(final_user_input_prompt) # ★ここを final_user_input_prompt に変更★

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        gemini_chat_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                gemini_chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                # レベル変更時のシステムメッセージは履歴に含めない
                if "Okay, switching to the " not in msg["content"]:
                    gemini_chat_history.append({"role": "model", "parts": [msg["content"]]})

        chat = model.start_chat(history=gemini_chat_history)

        try:
            response_generator = chat.send_message(final_user_input_prompt, stream=True) # ★ここを final_user_input_prompt に変更★

            for chunk in response_generator:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Assistantの返答を音声で再生
            if use_audio_io and _can_use_gcp_voice and full_response:
                audio_output = synthesize_text_gcp(full_response)
                if audio_output:
                    st.audio(audio_output, format="audio/mp4", autoplay=True)

        except Exception as e:
            st.error(f"An error occurred with Gemini: {e}. Please try again.")
            st.session_state.messages.append({"role": "assistant", "content": f"An error occurred with Gemini: {e}. Please try again."})