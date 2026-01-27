import streamlit as st
import time
import google.generativeai as genai
# import os # os はもう不要かもしれません

# 0. 環境変数の設定 (Streamlit Cloudでの推奨方法)
# `st.secrets` から API キーを取得
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("エラー: GEMINI_API_KEY が Streamlit Secrets に設定されていません。")
    st.stop() # アプリの実行を停止

genai.configure(api_key=api_key)

# 1. モデルの初期化 (成功した gemini-flash-latest を使用)
model = genai.GenerativeModel('gemini-flash-latest')

# チャット履歴をStreamlitのセッションステートで管理
# st.session_state はセッションごとにデータを保持するための辞書のようなものです。
if "messages" not in st.session_state:
    st.session_state.messages = [] # []は空のリスト

# キャラクタースペック (初回のみ表示)
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "みゃーお！吾輩は猫である。名前はまだない。\n君、吾輩と遊んでニャン？"})

# Streamlit UIの構築
st.title("吾輩は猫であるチャットボット")
st.write("「吾輩は猫である」の主人公視点で会話をする猫型AIです。")

# 既存のチャット履歴を表示
for message in st.session_state.messages:
    # Streamlitのチャットメッセージ表示コンポーネントを使用
    # role='user' なら右、role='assistant' なら左に表示されます
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

        # Streamlitのチャットセッションを開始（履歴をGeminiに渡す）
        # st.session_state.messages から Gemini が理解できる形式に変換
        # role mapping: user -> user, assistant -> model
        gemini_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                # アシスタントからの最初のメッセージ（キャラクタースペック）は履歴に含めない
                # または、Geminiが処理できる形式に変換する
                if msg["content"] != "みゃーお！吾輩は猫である。名前はまだない。\n君、吾輩と遊んでニャン？":
                     gemini_messages.append({"role": "model", "parts": [msg["content"]]})


        # 最新のユーザープロンプトをGeminiに送信
        # start_chatでこれまでの履歴を渡し、send_messageで最新のメッセージを送信
        # 最新のユーザープロンプトは gemini_messages にまだ含まれていないため、直接渡す
        # ただし、GeminiのAPIは `start_chat` で履歴を渡した後、後続の `send_message` には
        # 最新のユーザー入力のみを渡すのが基本です。
        # ここでは simplified で、履歴を毎回構築して新しいチャットを開始する形式にしています。
        # 厳密には `model.start_chat(history=...)` を利用し、そのチャットオブジェクトの `send_message` を使うべきですが、
        # Streamlitのステート管理では毎回新しいチャットオブジェクトを作っても問題ないことが多いです。

        # 履歴を毎回モデルに渡す（statefulにする）ために、以下のようにします。
        chat = model.start_chat(history=[
            {"role": "user", "parts": [m["content"]]} if m["role"] == "user" else {"role": "model", "parts": [m["content"]]}
            for m in st.session_state.messages[:-1] # 最新のユーザー入力以外を履歴として渡す
            if m["content"] != "みゃーお！吾輩は猫である。名前はまだない。\n君、吾輩と遊んでニャン？" # 初回メッセージを除外
        ])

        try:
            # 最新のユーザーの入力だけを send_message に渡す
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