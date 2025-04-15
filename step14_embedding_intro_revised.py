# step14_embedding_intro_revised.py
import os
import sys
from dotenv import load_dotenv # .env ファイルを使う場合に必要

# OpenAIEmbeddings をインポート
try:
    from langchain_openai import OpenAIEmbeddings
    print("OpenAIEmbeddings をインポートしました (from langchain_openai)。")
except ImportError:
    print("エラー: langchain-openai が見つかりません。")
    print("   'pip install -U langchain-openai' を実行してください。")
    sys.exit(1)

print("--- 必要なモジュールのインポート完了 ---")

# --- APIキーの準備 ---
# 方法1: 環境変数から直接読み込む (推奨)
openai_api_key = os.getenv("OPENAI_API_KEY")

# 方法2: .env ファイルから読み込む (環境変数が設定されていない場合)
if not openai_api_key:
    print("環境変数 OPENAI_API_KEY が未設定です。 .env ファイルからの読み込みを試みます。")
    if load_dotenv():
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            print(".env ファイルから API キーを読み込みました。")
        else:
            print("警告: .env ファイルに OPENAI_API_KEY が見つかりませんでした。")
    else:
        print("警告: .env ファイルが見つかりませんでした。")

# APIキーが最終的に設定されているか確認
if not openai_api_key:
    print("エラー: OpenAI API キーが設定されていません。")
    print("環境変数 または .env ファイルに OPENAI_API_KEY を設定してください。")
    sys.exit(1)
else:
    # セキュリティのため、キーの一部のみ表示 (オプション)
    print(f"OpenAI API キーが読み込まれました (キーの先頭: {openai_api_key[:5]}...)")
# step14_embedding_intro_revised.py (続き)

try:
    # モデル名と、オプションで次元数を指定して初期化
    # text-embedding-3-small が一般的だが、large も利用可能 (高性能/高コスト)
    # dimensions を指定しない場合、モデルのデフォルト次元数 (例: 1536) になる
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small", # モデル名を明示的に指定
        # dimensions=512, # 必要なら次元数を削減 (オプション)
        api_key=openai_api_key # 環境変数から読み込んだキーを渡す
    )
    print(f"\n--- OpenAIEmbeddings 初期化完了 ---")
    print(f"  使用モデル: {embeddings_model.model}")
    # dimensions を指定した場合、それが反映されるか確認 (指定しなければ None)
    print(f"  指定次元数: {embeddings_model.dimensions}")

except Exception as e:
    # APIキーが無効、ネットワークエラー、不正なモデル名などでエラーが発生する可能性
    print(f"エラー: OpenAIEmbeddings の初期化に失敗しました: {e}")
    print("   考えられる原因: APIキーが無効、モデル名が間違っている、ネットワーク接続など。")
    sys.exit(1)
# step14_embedding_intro_revised.py (続き)
print("\n--- 単一テキストのベクトル化 (.embed_query) ---")

query_text = "こんにちは、世界！"
print(f"入力テキスト: '{query_text}'")

try:
    query_vector = embeddings_model.embed_query(query_text)
    print("ベクトル化完了。")

    print(f"  ベクトル型: {type(query_vector)}")
    print(f"  ベクトル次元数: {len(query_vector)}") # 指定した dimensions またはデフォルト次元数
    print(f"  ベクトルの一部 (最初の5要素): {query_vector[:5]}...")

except Exception as e:
    # API呼び出し時のエラー (レート制限、接続エラーなど)
    print(f"エラー: embed_query の実行中にエラーが発生しました: {e}")
    print("   考えられる原因: APIキーの問題、ネットワーク接続、OpenAI API側の問題 (レート制限など)。")
# step14_embedding_intro_revised.py (続き)
print("\n--- 複数テキストの一括ベクトル化 (.embed_documents) ---")

document_texts = [
    "LangChain は LLM アプリケーション開発を支援します。",
    "テキストをベクトルに変換するのが Embedding です。",
    "ベクトル検索は RAG の重要な要素です。"
]
print(f"入力テキストリスト (計 {len(document_texts)} 件):")
for i, text in enumerate(document_texts):
    print(f"  {i+1}. '{text}'")

try:
    document_vectors = embeddings_model.embed_documents(document_texts)
    print("一括ベクトル化完了。")

    print(f"  結果の型: {type(document_vectors)}")
    print(f"  ベクトル数: {len(document_vectors)}")
    if document_vectors:
        print(f"  各ベクトルの次元数: {len(document_vectors[0])}")
        print(f"  最初のテキストのベクトル (最初の5要素): {document_vectors[0][:5]}...")
        print(f"  2番目のテキストのベクトル (最初の5要素): {document_vectors[1][:5]}...")
    else:
        print("  ベクトルリストが空です。")

except Exception as e:
    print(f"エラー: embed_documents の実行中にエラーが発生しました: {e}")
    print("   考えられる原因: APIキーの問題、ネットワーク接続、OpenAI API側の問題 (レート制限など)。")


print("\n--- 処理終了 ---")