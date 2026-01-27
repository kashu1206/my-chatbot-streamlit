import streamlit as st
import time
import google.generativeai as genai

# 0. 環境変数の設定 (Streamlit Cloudでの推奨方法)
# `st.secrets` から API キーを取得
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("エラー: GEMINI_API_KEY が Streamlit Secrets に設定されていません。")
    st.stop() # アプリの実行を停止

genai.configure(api_key=api_key)

# 1. モデルの初期化
# ここで `system_instruction` を追加し、キャラクター設定をGeminiモデルに常に指示します。
model = genai.GenerativeModel(
    'gemini-flash-latest',
    system_instruction="あなたは夏目漱石の「吾輩は猫である」の主人公の猫です。一人称は「吾輩」で、語尾は「ニャン」です。人間を観察する視点から、少し生意気でおっとりとした口調で会話してください。"
)

# チャット履歴をStreamlitのセッションステートで管理
if "messages" not in st.session_state:
    st.session_state.messages = []

    # 初回表示用のキャラクターメッセージ。これはUIに表示するためのみです。
    # 上記の system_instruction でキャラクター設定はGeminiに伝えられているため、
    # この最初のメッセージをGeminiのチャット履歴に含める必要はありません。
    st.session_state.messages.append({"role": "assistant", "content": "みゃーお！吾輩は猫である。名前はまだない。\n君、吾輩と遊んでニャン？"})

# Streamlit UIの構築
st.title("吾輩は猫であるチャットボット")
st.write("「吾輩は猫である」の主人公視点で会話をする猫型AIです。")

# 既存のチャット履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの入力を受け付ける
if prompt := st.chat_input("吾輩に話しかけるニャン..."): # ユーザーが何か入力したら
    # ユーザーのメッセージを履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gemini APIにメッセージを送信し、応答を取得
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # 返信を逐次表示するためのプレースホルダ
        full_response = ""

        # Gemini APIに渡すためのチャット履歴を構築
        # system_instruction を使用しているため、st.session_state.messages の最初の
        # アシスタントからの初期メッセージは Gemini のチャット履歴には含めません。
        # 実際のユーザーと AI の会話履歴のみを start_chat に渡します。
        gemini_chat_history = []
        # st.session_state.messages は `[初期アシスタント, ユーザー1, アシスタント1, ..., ユーザーN]`
        # の形式で格納されています。
        # `history` に含めるのは `ユーザー1, アシスタント1, ..., アシスタントN-1` までです。
        # `prompt` は `ユーザーN` に相当し、これは `send_message` で直接渡します。
        # そのため、`st.session_state.messages` の最初の要素と最後の要素は除外します。
        for msg in st.session_state.messages[1:-1]: # 最初の初期メッセージと最新のユーザープロンプトを除外
            if msg["role"] == "user":
                gemini_chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_chat_history.append({"role": "model", "parts": [msg["content"]]})

        # これまでの会話履歴（gemini_chat_history）を使ってチャットを開始
        chat = model.start_chat(history=gemini_chat_history)

        try:
            # 最新のユーザーからの入力（prompt）だけを send_message に渡して応答を取得
            response = chat.send_message(prompt, stream=True)

            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌") # カーソル風の表示
                time.sleep(0.05) # 少し間隔を空ける
            message_placeholder.markdown(full_response) # 最終的なレスポンス

            # アシスタントのメッセージを履歴に追加
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"エラーが発生したニャン...: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"エラーが発生したニャン...: {e}"})