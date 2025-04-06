# step2_hello_llm.py

import os
from dotenv import load_dotenv
# OpenAIモデルと連携するためのChatOpenAIをインポート
from langchain_openai import ChatOpenAI

# 1. APIキーの読み込み準備
# .envファイルから環境変数を読み込む（これによりChatOpenAIがAPIキーを認識できる）
load_dotenv()

print("--- LLMとの対話を開始します ---")

# 2. LLM (ChatOpenAI) の初期化
# まずは推奨される "gpt-3.5-turbo" を使ってみましょう
# temperature は応答の多様性を制御するパラメータです
try:
    # モデルを指定。temperatureはお好みで調整（0に近いほど固い応答、高いほど多様）
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    print(f"✅ OK: ChatOpenAI ({llm.model_name}) の準備ができました。")
except Exception as e:
    print(f"❌ エラー: ChatOpenAI の初期化に失敗しました: {e}")
    print("   確認: APIキーは正しく設定されていますか？ OpenAIアカウントは有効ですか？")
    exit() # 続行不可

# 3. LLMへの質問 (プロンプト) を準備
prompt = "こんにちは！ あなたは誰ですか？"
print(f"\n> あなたの質問: {prompt}")

# 4. LLMへの質問を実行 (invoke)
# .invoke() を使ってプロンプトをLLMに送り、応答を待ちます
try:
    response = llm.invoke(prompt)
    print("✅ OK: LLMからの応答を受信しました。")
except Exception as e:
    print(f"❌ エラー: LLMへの問い合わせ中にエラーが発生しました: {e}")
    print("   確認: ネットワーク接続、APIキー、OpenAIアカウント利用状況などを確認してください。")
    exit() # 続行不可

# 5. 応答の表示
# 応答は AIMessage オブジェクトで返ってきます。
# まずはオブジェクト全体を見てみましょう。
print("\n--- 応答オブジェクト全体 (AIMessage) ---")
print(response)

# AIMessageオブジェクトから実際の応答テキスト(content)を取り出して表示します。
print("\n--- AIからの応答 (content) ---")
print(response.content)

print("\n--- LLMとの対話が終了しました ---")