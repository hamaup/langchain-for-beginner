# step5_lcel_pipe_intro.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# StrOutputParser を langchain_core からインポート
from langchain_core.output_parsers import StrOutputParser

# 環境変数の読み込み
load_dotenv()
print("--- 環境変数読み込み完了 ---")

# LLMの準備
try:
    # temperature=0 に設定し、応答の再現性を高める
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    print(f"--- LLM準備完了: {llm.model_name} (temperature={llm.temperature}) ---")
    print("   (temperature=0 は、毎回ほぼ同じ応答を返す設定です)")
except Exception as e:
    # APIキーがない、無効な場合などにエラーが発生する可能性があります
    print(f"❌ エラー: ChatOpenAI の初期化に失敗しました: {e}")
    print("   確認: OpenAI APIキーは正しく .env ファイルに設定されていますか？")
    exit() # 処理を中断

# プロンプトテンプレートの準備
# {country} という名前の変数（プレースホルダ）を含むテンプレート
# よりシンプルな質問応答タスクに変更
prompt = ChatPromptTemplate.from_template(
    "{country} の首都はどこですか？"
)
print("--- プロンプトテンプレート準備完了 ---")

# 出力パーサーの準備
output_parser = StrOutputParser()
print("--- Output Parser (StrOutputParser) 準備完了 ---")


# --- チェーン 1: プロンプト | LLM ---
print("\n--- チェーン 1: プロンプト | LLM ---")
chain_prompt_llm = prompt | llm
print("   チェーン構築完了")

# プロンプトテンプレート内の {country} に代入する値を定義
# これがチェーンの最初の入力となる
# キー名はテンプレート内の変数名 {country} と一致させる
input_data = {"country": "フランス"}

try:
    print(f"\n   実行中... チェーンへの入力 (辞書形式): {input_data}")
    # チェーンの .invoke() には、最初の要素 (この場合はプロンプト) が
    # 必要とする入力形式（ここでは辞書）でデータを渡す
    response_message = chain_prompt_llm.invoke(input_data)
    print("✅ OK: 応答を受信しました。")

    print("\n   【応答の型】:", type(response_message))
    print("   【応答 (AIMessage)】:")
    print("   ", response_message)
    print("\n   【応答の内容 (content)】:")
    print("   ", response_message.content)
except Exception as e:
    print(f"❌ エラー: チェーン 1 の実行中にエラーが発生しました: {e}")


# --- チェーン 2: プロンプト | LLM | StrOutputParser ---
print("\n--- チェーン 2: プロンプト | LLM | StrOutputParser ---")
chain_full = prompt | llm | output_parser
print("   チェーン構築完了")

try:
    # チェーン1と同じ入力データを使用する
    print(f"\n   実行中... チェーンへの入力 (辞書形式): {input_data}")
    # こちらのチェーンも、最初の要素 (プロンプト) が必要とする辞書を入力とする
    response_string = chain_full.invoke(input_data)
    print("✅ OK: 応答を受信しました。")

    print("\n   【応答の型】:", type(response_string))
    print("   【応答 (文字列)】:")
    print("   ", response_string)
except Exception as e:
    # LLMの応答形式が予期せぬものだった場合などにエラーになる可能性もゼロではない
    print(f"❌ エラー: チェーン 2 の実行中にエラーが発生しました: {e}")


print("\n--- 処理終了 ---")