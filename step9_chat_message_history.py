# step9_chat_message_history.py
import os
import sys # 標準的な終了処理のため
from dotenv import load_dotenv

# ChatMessageHistory をインポート
try:
    from langchain_community.chat_message_histories import ChatMessageHistory
    print("ChatMessageHistory をインポートしました。")
except ImportError:
    print("エラー: langchain-community が見つかりません。")
    print("   'pip install langchain-community' を実行してください。")
    sys.exit(1) # エラー終了

# メッセージオブジェクトの型を確認・利用するためにインポート
from langchain_core.messages import HumanMessage, AIMessage

print("--- 必要なモジュールのインポート完了 ---")

# (補足) .env ファイルは現時点では使いませんが、設定を読み込むコードを入れておきます
load_dotenv()

# ChatMessageHistory オブジェクトを作成
history = ChatMessageHistory()
print("--- ChatMessageHistory オブジェクト作成完了 ---")
print(f"初期状態の履歴 (.messages): {history.messages}")

# メッセージを追加
print("\n--- メッセージを追加します ---")
# 方法1: 文字列でユーザーとAIの発言を追加
history.add_user_message("LangChainについて基本的なことを教えて。")
print("ユーザーメッセージを文字列で追加しました。")
history.add_ai_message("LangChainはLLMアプリ開発を助けるフレームワークですよ。")
print("AIメッセージを文字列で追加しました。")

# 方法2: メッセージオブジェクトを直接追加
human_msg_turn2 = HumanMessage(content="もう少し詳しく知りたいな。")
history.add_message(human_msg_turn2)
print("ユーザーメッセージをオブジェクトで追加しました。")
ai_msg_turn2 = AIMessage(content="コンポーネントを組み合わせて複雑な処理を作れます。")
history.add_message(ai_msg_turn2)
print("AIメッセージをオブジェクトで追加しました。")

# 現在の履歴を確認
print("\n--- 現在の履歴を確認します ---")
current_messages = history.messages
print(f"取得した履歴はリスト形式です: {type(current_messages)}")
print(f"現在の履歴の長さ: {len(current_messages)}") # 追加したメッセージの数

print("\n履歴の内容:")
# 取得したリストをループして、各メッセージの型と内容を表示
for i, msg in enumerate(current_messages):
    print(f"  {i+1}. 型: {type(msg).__name__}, 内容: '{msg.content}'")

# 履歴をクリア（オプション）
print("\n--- 履歴をクリアします ---")
history.clear()
print("履歴をクリアしました。")
print(f"クリア後の履歴 (.messages): {history.messages}") # 空のリストが表示されるはず

print("\n--- 処理終了 ---")