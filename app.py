import streamlit as st
import time
import google.generativeai as genai
import io
import os
import json
import base64
import random

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
        st.sidebar.success("GCP Credentials loaded successfully from Base64 secret!")

    # 2. 直接JSON文字列として設定された認証情報
    elif "GCP_CREDENTIALS" in st.secrets:
        raw_credentials = st.secrets["GCP_CREDENTIALS"]
        if isinstance(raw_credentials, dict):
            _decoded_gcp_credentials_json_string = json.dumps(raw_credentials)
        else:
            _decoded_gcp_credentials_json_string = raw_credentials
        st.sidebar.success("GCP Credentials loaded successfully from direct secret!")
    
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

        trimmed_audio = trimmed_audio.set_sample_width(2)

        output_buffer = io.BytesIO()
        trimmed_audio.export(output_buffer, format="wav")
        trimmed_audio_bytes = output_buffer.getvalue()

        audio = speech.RecognitionAudio(content=trimmed_audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US", # 英語会話パートナーなのでen-US
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
def synthesize_text_gcp(text, voice_name="en-US-Standard-F", ssml_gender="FEMALE"):
    # クライアントが初期化されているか再確認
    if _tts_client is None:
        st.error("Text-to-Speech client is not initialized. Cannot synthesize speech.")
        return None

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice_name,
            ssml_gender=texttospeech.SsmlVoiceGender[ssml_gender.upper()] # 文字列からEnumに変換
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3, # MP3に統一
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
    # system_instruction を使うため、gemini-1.5-flash-latest を使用
    return genai.GenerativeModel('gemini-1.5-flash-latest')
model = get_gemini_model()

# --- キャラクターの定義 ---
# 各キャラクターの基礎となる指示
# ★変更点: 他のキャラクターの発言を考慮する指示を追加★
BASE_INSTRUCTION = (
    "You are an English conversation partner who helps users improve their English skills. "
    "You are also an experienced English teacher with extensive experience guiding native Japanese speakers in learning English as a foreign language. "
    "Please keep in mind that the user is a native Japanese speaker throughout your interactions. "
    "**Always respond only in English. Do not use Japanese at all.**"
    "**Important: You are participating in a group conversation with other characters. If another character has just spoken, please take their statement into account and build upon it, clarify, or offer a different perspective. Make your response flow naturally within the group dialogue.**" 
)

# キャラクターごとの詳細設定
CHARACTERS = {
    "Hana": {
        "name": "Hana",
        "icon": "🌸", # Junior high school student
        "voice_name": "en-US-Standard-F", # Female voice from GCP TTS
        "ssml_gender": "FEMALE",
        "persona": BASE_INSTRUCTION + (
            " Your name is Tanaka Hana. You are a girl from Wakaba Junior High School, originally from Wakaba City."
            " You have a gentle and meticulous personality, and your friends often consult you when they're in trouble."
            " You've been dedicated to soccer since age 3. Recently, you've been enjoying family camping trips and mastering camp cooking."
            " You're preparing to play in an overseas soccer league after junior high school graduation."
            " Always speak in a friendly, gentle, and encouraging manner, appropriate for a junior high school student helping a friend learn English."
        ),
    },
    "Mark": {
        "name": "Mark",
        "icon": "👨‍👦", # Father figure
        "voice_name": "en-US-Standard-D", # Male voice from GCP TTS
        "ssml_gender": "MALE",
        "persona": BASE_INSTRUCTION + (
            " Your name is Mark. You are a father from the United States with a cheerful and humorous personality."
            " You are very proud of your children and love spending time with your family."
            " You work as a software engineer and enjoy discussing technology and current events."
            " You also enjoy outdoor activities and traveling."
            " Always speak in a cheerful, friendly, and slightly humorous tone, like a supportive and engaging father figure."
        ),
    },
    "Ms. Brown": {
        "name": "Ms. Brown",
        "icon": "👩‍🏫", # Teacher
        "voice_name": "en-US-Wavenet-C", # Female voice, professional tone from GCP TTS
        "ssml_gender": "FEMALE",
        "persona": BASE_INSTRUCTION + (
            " Your name is Ms. Brown. You are a highly experienced and professional English teacher."
            " You are known for your clear explanations, structured lessons, and ability to identify areas for improvement."
            " You are patient, supportive, and always provide constructive feedback."
            " You have a deep understanding of common difficulties faced by Japanese English learners."
            " Always speak in a clear, professional, polite, and encouraging manner, like a seasoned English teacher."
        ),
    },
}

AVAILABLE_CHARACTER_NAMES = list(CHARACTERS.keys())

# --- Streamlit UI ---
st.title("English Conversation Partner")

# サイドバーにキャラクター選択
selected_character_key = st.sidebar.selectbox(
    "Select your conversation partner:",
    ["Anyone"] + AVAILABLE_CHARACTER_NAMES,
    index=0 # Default to "Anyone"
)

# 音声入出力の設定
use_audio_io = st.sidebar.checkbox("Enable voice input/output", value=True)


# --- Chat Historyの初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 初期の挨拶 (Systemとして)
    st.session_state.messages.append({
        "role": "assistant",
        "name": "System",
        "icon": "🗣️",
        "content": "Hello! I'm your English conversation partner. Which character would you like to speak with today, or should anyone respond? I will only respond in English.",
    })
    # 各キャラクターの個別対話履歴を初期化
    st.session_state.char_dialogue_histories = {name: [] for name in AVAILABLE_CHARACTER_NAMES}


# --- Chat Historyの表示 ---
# messages リストが更新されるたびに再描画される
for message in st.session_state.messages:
    # メッセージに `icon` キーがない場合のデフォルトアイコンを設定
    avatar = message.get("icon", "👤" if message["role"] == "user" else "🤖")
    with st.chat_message(message["role"], avatar=avatar):
        # `name` キーがある場合は名前も表示
        if message.get("name"):
            st.markdown(f"**{message['name']}:** {message['content']}")
        else:
            st.markdown(message["content"])


# --- 音声入力 (mic_recorder) ---
if use_audio_io and mic_recorder:
    audio_bytes = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", key="recorder")
    if audio_bytes:
        with st.spinner("Transcribing audio..."):
            transcribed_text = transcribe_audio_gcp(audio_bytes)
        if transcribed_text:
            st.session_state.messages.append({"role": "user", "icon": "👤", "content": transcribed_text})
            st.rerun()


# --- LLMからの応答生成 ---
# chat_inputが呼ばれたら処理を開始
if user_input := st.chat_input("Type your message here..."):
    # ユーザーメッセージを全体の履歴に追加
    st.session_state.messages.append({"role": "user", "icon": "👤", "content": user_input})

    # 応答するキャラクターのリストを決定
    responding_character_keys = []
    if selected_character_key == "Anyone":
        # 応答するキャラクターの数をランダムに決定 (1人、2人、または3人)
        num_responders = random.randint(1, min(len(AVAILABLE_CHARACTER_NAMES), 3)) # 最大3人、または利用可能な全キャラクター数
        responding_character_keys = random.sample(AVAILABLE_CHARACTER_NAMES, k=num_responders)
        random.shuffle(responding_character_keys) # 選ばれたキャラクターの応答順序もランダムに
    else:
        # 特定のキャラクターが選択された場合
        responding_character_keys = [selected_character_key]

    # ★追加: 前のキャラクターの応答を格納する変数 (今回のセッション内限定のコンテキスト) ★
    context_from_previous_char_response = ""
    previous_char_name_for_context = ""

    # 各応答キャラクターに対して処理
    for char_key in responding_character_keys:
        char_info = CHARACTERS[char_key]
        
        # このキャラクターの個別履歴を取得
        char_history = st.session_state.char_dialogue_histories[char_key]

        # LLMセッションの開始 (毎回新しいセッションを作成することで、system_instruction と history を正確に設定)
        cleaned_history_for_gemini = []
        for msg in char_history:
            if msg["role"] == "user":
                cleaned_history_for_gemini.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "model":
                cleaned_history_for_gemini.append({"role": "model", "parts": [msg["content"]]})

        chat_session = model.start_chat(
            history=cleaned_history_for_gemini,
            system_instruction=char_info["persona"]
        )
        
        # ★変更点: 他のキャラクターの発言をユーザー入力として追記★
        user_input_for_llm = user_input
        if context_from_previous_char_response:
            user_input_for_llm += (
                f"\n\n(Context from recent conversation: The previous character, {previous_char_name_for_context}, "
                f"just said to the user: '{context_from_previous_char_response}'. Please consider this in your response.)"
            )

        with st.chat_message("assistant", avatar=char_info["icon"]):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # ユーザー入力をセッションに送信し、ストリーミング応答を取得
                response_stream = chat_session.send_message(user_input_for_llm, stream=True)
                
                for chunk in response_stream:
                    full_response += chunk.text
                    # 応答中にキャラクター名を表示し、カーソルエフェクトを付ける
                    message_placeholder.markdown(f"**{char_info['name']}:** {full_response}▌")
                
                # 最終的な表示 (カーソルエフェクト削除)
                message_placeholder.markdown(f"**{char_info['name']}:** {full_response}")
                
                if full_response:
                    # 全体の履歴に追加
                    st.session_state.messages.append({
                        "role": "assistant",
                        "name": char_info["name"],
                        "icon": char_info["icon"],
                        "content": full_response,
                    })

                    # 個別キャラクターの履歴に追加 (ユーザー入力とAI応答)
                    # ここに他のキャラクターの発言は含めない
                    st.session_state.char_dialogue_histories[char_key].append({
                        "role": "user", "content": user_input # 元のユーザー入力のみを記録
                    })
                    st.session_state.char_dialogue_histories[char_key].append({
                        "role": "model", "content": full_response
                    })

                    # ★追加: 次のキャラクターのために現在の応答をコンテキストとして保持★
                    context_from_previous_char_response = full_response
                    previous_char_name_for_context = char_info["name"]

                    # 音声出力
                    if use_audio_io and _can_use_gcp_voice:
                        audio_output = synthesize_text_gcp(
                            full_response,
                            voice_name=char_info["voice_name"],
                            ssml_gender=char_info["ssml_gender"]
                        )
                        if audio_output:
                            st.audio(audio_output, format="audio/mpeg")
                            st.markdown("🔊 *(Note: On iOS, please tap the play button manually.)*")
                else:
                    st.warning(f"{char_info['name']} did not generate a response.")

            except Exception as e:
                st.error(f"Error generating response from Gemini for {char_info['name']}: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "name": char_info["name"],
                    "icon": char_info["icon"],
                    "content": "I apologize, but I could not generate a response. Please try again or ask a different question.",
                })
    
    st.rerun() # すべての応答が完了したら一度だけリロード