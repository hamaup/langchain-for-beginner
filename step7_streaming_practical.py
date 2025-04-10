# step7_streaming_practical.py
import os
import asyncio
from dotenv import load_dotenv

# --- LangChain Core Modules ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- LangChain OpenAI Integration ---
from langchain_openai import ChatOpenAI

# 環境変数の読み込み
load_dotenv()
print("--- 環境変数読み込み完了 ---")

# LLMの準備 (streaming=True を指定)
try:
    # streaming=True を指定して、ストリーミングに適した応答を促す
    # temperature を少し上げて多様な応答を生成させる
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, streaming=True)
    print(f"--- LLM準備完了: {llm.model_name} (temperature={llm.temperature}, streaming=True) ---")
except Exception as e:
    print(f"❌ エラー: ChatOpenAI の初期化に失敗しました: {e}")
    print("   (APIキー、アカウント設定、ライブラリのバージョンを確認してください)")
    exit()

# プロンプトテンプレートの準備
prompt = ChatPromptTemplate.from_template(
    "{topic} について、その魅力と将来性を３つのポイントで解説してください。"
)
print("--- プロンプトテンプレート準備完了 ---")

# 出力パーサーの準備 (LLM応答を文字列に変換)
output_parser = StrOutputParser()
print("--- Output Parser (StrOutputParser) 準備完了 ---")

# LCEL チェーンの構築
chain = prompt | llm | output_parser
print("--- LCEL チェーン構築完了 ---")

# --- 同期ストリーミング (.stream) ---
print("\n--- 同期ストリーミング (.stream) 開始 ---")
topic_sync = "再生可能エネルギー"
print(f"> トピック: {topic_sync}")
print("AI応答:")
try:
    full_response_sync = ""
    # .stream() は同期イテレータを返す
    for chunk in chain.stream({"topic": topic_sync}):
        # 受け取ったチャンクを改行せずに出力 (flush=Trueで即時表示)
        print(chunk, end="", flush=True)
        full_response_sync += chunk # 完全な応答を後で確認するために結合
    print("\n--- .stream() 完了 ---")
    # print("\n[デバッグ] 結合された応答(同期):", full_response_sync) # 必要に応じて確認
except Exception as e:
    print(f"\n❌ エラー: 同期ストリーミング中にエラーが発生しました: {e}")
    print("   (ネットワーク接続、APIキー、利用制限などを確認してください)")


# --- 非同期ストリーミング (.astream) ---
# 非同期ストリーミング用の関数を定義
async def run_async_streaming():
    print("\n--- 非同期ストリーミング (.astream) 開始 ---")
    topic_async = "人工知能と創造性"
    print(f"> トピック: {topic_async}")
    print("AI応答:")
    try:
        full_response_async = ""
        # .astream() は非同期ジェネレータを返す
        async for chunk in chain.astream({"topic": topic_async}):
            # 受け取ったチャンクを改行せずに出力
            print(chunk, end="", flush=True)
            full_response_async += chunk # 完全な応答を後で確認するために結合
        print("\n--- .astream() 完了 ---")
        # print("\n[デバッグ] 結合された応答(非同期):", full_response_async) # 必要に応じて確認
    except Exception as e:
        print(f"\n❌ エラー: 非同期ストリーミング中にエラーが発生しました: {e}")
        print("   (ネットワーク接続、APIキー、利用制限などを確認してください)")

# スクリプトとして直接実行された場合に非同期関数を呼び出す
if __name__ == "__main__":
    print("\n--- 非同期処理実行 ---")
    try:
        # asyncio.run() を使って非同期関数を実行
        asyncio.run(run_async_streaming())
    except RuntimeError as e:
        # Jupyter Notebook など、既にイベントループが実行中の環境への対応
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                 print("\n注意: 既存のイベントループが検出されました。")
                 print("      Jupyter Notebookなどの環境では、セルで `await run_async_streaming()` を直接実行してください。")
            else:
                 print(f"イベントループは存在しますが、実行中ではありません。: {e}")
        except RuntimeError:
             print(f"実行中のイベントループが見つかりませんでした。: {e}")
    except Exception as e:
        print(f"非同期処理の実行中に予期せぬエラーが発生しました: {e}")


print("\n--- 全ての処理が終了しました ---")