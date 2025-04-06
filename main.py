import os
from dotenv import load_dotenv
print("--- 環境チェック開始 ---")

# 1. ライブラリのインポート確認
try:
    from langchain_openai import ChatOpenAI
    print("✅ OK: langchain-openai ライブラリが見つかりました。")
except ImportError:
    print("❌ エラー: langchain-openai ライブラリが見つかりません。")
    print("   確認: 仮想環境は有効ですか？ 'pip install langchain-openai' は実行しましたか？")
    exit() # 続行不可

# 2. .envファイルの読み込み確認
#    load_dotenv() は .env ファイルを探して環境変数に読み込みます
#    見つからなくてもエラーにはなりませんが、後のAPIキー読み込みで問題が出ます
env_found = load_dotenv()
if env_found:
    print("✅ OK: .env ファイルが見つかり、読み込みを試みました。")
else:
    print("⚠️ 注意: .env ファイルが見つかりません。作業フォルダ直下にありますか？")
    # 実行は続行してみる

# 3. APIキーの読み込み確認
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("✅ OK: 環境変数 'OPENAI_API_KEY' からAPIキーを読み込めました。")
else:
    print("❌ エラー: 環境変数 'OPENAI_API_KEY' が設定されていません。")
    print("   確認: .env ファイルに正しく記述されていますか？ (例: OPENAI_API_KEY='sk-...')")
    # APIキーがないとこの先で失敗する可能性が高いが、一旦続行してみる

# 4. ChatOpenAIの初期化試行 (APIコールはまだしない)
if api_key: # APIキーがある場合のみ試行
    try:
        # ここで初めて ChatOpenAI クラスを使う際に API キーが内部的に検証されることがあります
        llm = ChatOpenAI()
        # 実際にAPIコールするわけではないので、モデル名はデフォルトが表示されるはず
        print(f"✅ OK: ChatOpenAI クラスの準備ができました (デフォルトモデル: {llm.model_name})。")
    except Exception as e:
        print(f"❌ エラー: ChatOpenAI クラスの準備中に問題が発生しました: {e}")
        print("   確認: APIキーは有効ですか？ OpenAIアカウントの支払い設定は有効ですか？")
else:
    print("⚠️ 注意: APIキーがないため、ChatOpenAI クラスの準備はスキップします。")


print("--- 環境チェック終了 ---")