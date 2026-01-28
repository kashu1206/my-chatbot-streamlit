import base64 # <-- これが以前インポートされていなかった問題に対する修正（今回は既に修正済みと仮定）
import streamlit as st
import time
import google.generativeai as genai
import io
import os
import json
import tempfile

# GCP Speech-to-Text and Text-to-Speech clients
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1 as texttospeech
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

# --- GCP Credentials (for Speech-to-Text and Text-to-Speech) ---
_can_use_gcp_voice = False
_speech_client = None
_texttospeech_client = None
_decoded_gcp_credentials_json_string = None 
_temp_key_file_path = None # 一時ファイルのパス

try:
    # 優先: Base64エンコードされた認証情報を確認
    if "GCP_CREDENTIALS_BASE64" in st.secrets:
        encoded_credentials = st.secrets["GCP_CREDENTIALS_BASE64"]
        _decoded_gcp_credentials_json_string = base64.b64decode(encoded_credentials.encode("utf-8")).decode("utf-8")
        st.success("GCP Credentials loaded successfully from Base64 secret!")

    # フォールバック: 直接GCP_CREDENTIALSがJSON文字列として設定されている場合
    elif "GCP_CREDENTIALS" in st.secrets:
        raw_credentials = st.secrets["GCP_CREDENTIALS"]
        if isinstance(raw_credentials, dict): # 既に辞書型の場合
            _decoded_gcp_credentials_json_string = json.dumps(raw_credentials)
        else: # 文字列の場合
            _decoded_gcp_credentials_json_string = raw_credentials
        st.success("GCP Credentials loaded successfully from direct secret!")
    
    else:
        st.warning("Warning: GCP_CREDENTIALS (Base64 or direct) for Speech-to-Text/Text-to-Speech are not set in Streamlit Secrets. Voice input/output will not be available.")
        # GCP認証情報がない場合でもGemini部分は動くように st.stop() は呼ばない
        
    # 認証情報文字列が存在する場合のみ、GCPクライアントを初期化
    if _decoded_gcp_credentials_json_string:
        # サービスアカウント情報で認証情報を生成
        _gcp_credentials_info = json.loads(_decoded_gcp_credentials_json_string)
        credentials = service_account.Credentials.from_service_account_info(_gcp_credentials_info)
        
        # 一時ファイルとしてJSONを保存し、GOOGLE_APPLICATION_CREDENTIALS 環境変数を設定
        # これは、`service_account.Credentials` を直接渡す場合でも、
        # 他のGoogle Cloudライブラリが環境変数を参照する可能性に備えるためです。
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as temp_key_file:
            temp_key_file.write(_decoded_gcp_credentials_json_string)
            _temp_key_file_path = temp_key_file.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _temp_key_file_path

        # GCPクライアントを初期化
        _speech_client = speech.SpeechClient(credentials=credentials)
        _texttospeech_client = texttospeech.TextToSpeechClient(credentials=credentials)
        _can_use_gcp_voice = True
        st.info("GCP Speech-to-Text and Text-to-Speech clients initialized.")

except Exception as e:
    st.error(f"Critical error during GCP credentials setup: {e}")
    st.warning("Voice input/output will not be available due to GCP client initialization failure.")
    _can_use_gcp_voice = False

# --- アプリ終了時に一時ファイルをクリーンアップ ---
# Streamlitのライフサイクルとtempfileの削除タイミングは複雑ですが、
# Streamlit Cloudではアプリの再起動時にファイルシステムがクリーンアップされるため、
# 明示的な os.remove(_temp_key_file_path) は必須ではないかもしれません。
# ローカルで実行していて、確実に削除したい場合は考慮します。

# --- 音声処理 (無音検出・トリミング) の設定 ---
SAMPLE_RATE = 16000  # Streamlit mic recorder は通常16kHzで録音される (GCP Speech-to-Textの推奨)

# --- 音声からテキストへ (Speech-to-Text) ---
def transcribe_audio_gcp(audio_bytes):
    if not _speech_client:
        st.error("Speech-to-Text client is not initialized.")
        return ""

    try:
        # pydubでオーディオバイトをロード (streamlit-mic-recorderは通常webm形式で出力)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        
        # 16kHz, 1チャンネルに変換 (GCP Speech-to-Textの推奨)
        if audio_segment.frame_rate != SAMPLE_RATE or audio_segment.channels != 1:
            audio_segment = audio_segment.set_frame_rate(SAMPLE_RATE).set_channels(1)
        
        # pydub.silence.detect_nonsilent を使用して無音区間を検出・トリミング
        nonsilent_chunks = detect_nonsilent(audio_segment, 
                                            min_silence_len=500, # 500ms以上の無音を検出
                                            silence_thresh=-35)  # -35dBFS以下の音量を無音と判定

        if not nonsilent_chunks: # 音声が全く検出されなかった場合
            st.info("No substantial speech detected after trimming.")
            return ""

        # 無音でないチャンクのみを結合
        trimmed_audio = AudioSegment.empty()
        for start_ms, end_ms in nonsilent_chunks:
            trimmed_audio += audio_segment[start_ms:end_ms]

        st.info(f"Original audio duration: {len(audio_segment)/1000:.2f}s, Trimmed audio duration: {len(trimmed_audio)/1000:.2f}s")

        # --- ここから修正 ---
        # pydub.AudioSegment のサンプル幅を16-bit (2バイト) に設定
        # GCP Speech-to-Text が LINEAR16 (16-bit PCM) を要求するため
        trimmed_audio = trimmed_audio.set_sample_width(2) # 2 bytes = 16 bits
        # --- 修正ここまで ---

        # 再びバイト列に変換 (WAV形式でヘッダを付与して送るのが最も確実)
        trimmed_audio_bytes = trimmed_audio.export(format="wav").read()

        audio = speech.RecognitionAudio(content=trimmed_audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # WAVなのでLINEAR16
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US", # 英語で認識
            enable_automatic_punctuation=True, # 自動句読点
        )

        response = _speech_client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        return transcript
    except Exception as e:
        st.error(f"Error transcribing audio with Google Cloud Speech-to-Text API: {e}")
        return ""

# --- テキストから音声へ (Text-to-Speech) ---
def synthesize_text_gcp(text):
    if not _texttospeech_client:
        st.error("Text-to-Speech client is not initialized.")
        return None

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", # 英語音声
            name="en-US-Standard-F", # 標準的な女性の声 (必要に応じて"en-US-Wavenet-F"なども検討)
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0, # 話速 (1.0が標準)
        )

        response = _texttospeech_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content # MP3バイト列
    except Exception as e:
        st.error(f"Error synthesizing speech with Google Cloud Text-to-Speech API: {e}")
        return None

# --- Geminiモデルの初期化 ---
@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel('gemini-pro')
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

    # 音声入力/出力のON/OFFトグル (GCPクライアントが初期化できた場合のみ有効)
    use_audio_io = st.toggle("Enable Voice Input/Output (GCP)", value=False, key="audio_io_toggle", disabled=not _can_use_gcp_voice)

    if use_audio_io and (mic_recorder is None or not _can_use_gcp_voice):
        st.warning("音声入出力は、`streamlit-mic-recorder` ライブラリが不足しているか、GCP認証情報が正しく設定されていないため無効です。")
    elif not use_audio_io:
        st.info("音声入出力は現在無効です。設定で有効にできます。")

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
            st.audio(audio_output, format="audio/mp3", autoplay=True)


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
            st.audio(audio_output, format="audio/mp3", autoplay=True)

# --- 既存のチャット履歴を表示 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ユーザーからの入力を受け付ける ---
user_input_prompt = ""
if use_audio_io:
    st.write("Click the mic and speak!")
    audio_bytes = None
    if mic_recorder: # mic_recorder が利用可能かチェック
        recorded_audio = mic_recorder(
            start_prompt="🎤 Start recording",
            stop_prompt="⏹️ Stop recording",
            just_once=True,
            use_container_width=True,
            key='user_mic_input'
        )
        if recorded_audio:
            audio_bytes = recorded_audio['bytes']

    if audio_bytes and _can_use_gcp_voice:
        with st.spinner("Processing audio and transcribing..."):
            user_input_prompt = transcribe_audio_gcp(audio_bytes)
            if user_input_prompt:
                st.write(f"You said: {user_input_prompt}")
            else:
                st.warning("Could not transcribe audio. Please try speaking clearer.")
    
    # 音声入力が成功しなかった場合、または音声入力が無効な場合はテキスト入力フォームを表示
    if not user_input_prompt: # user_input_prompt が空の場合
        user_input_prompt = st.chat_input("Start practicing English with me! (Or use mic above)", disabled=bool(audio_bytes))

else: # use_audio_io が False の場合
    user_input_prompt = st.chat_input("Start practicing English with me! (Type here)")


if user_input_prompt:
    st.session_state.messages.append({"role": "user", "content": user_input_prompt})
    with st.chat_message("user"):
        st.markdown(user_input_prompt)

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
            response_generator = chat.send_message(user_input_prompt, stream=True)

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
                    st.audio(audio_output, format="audio/mp3", autoplay=True)

        except Exception as e:
            st.error(f"An error occurred with Gemini: {e}. Please try again.")
            st.session_state.messages.append({"role": "assistant", "content": f"An error occurred with Gemini: {e}. Please try again."})