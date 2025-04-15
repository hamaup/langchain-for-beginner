# step13_text_splitters_revised.py
import os
import sys
from dotenv import load_dotenv
import tiktoken # トークン数を計算するために必要

# RecursiveCharacterTextSplitter をインポート
# 最新の推奨パッケージからインポート
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("RecursiveCharacterTextSplitter をインポートしました (from langchain_text_splitters)。")
    # 必要ならインストール: pip install langchain-text-splitters
except ImportError:
    print("エラー: langchain-text-splitters が見つかりません。")
    print("   'pip install langchain-text-splitters' を実行してください。")
    sys.exit(1)

# Document オブジェクトを扱うためにインポート (後で使用)
from langchain_core.documents import Document

print("--- 必要なモジュールのインポート完了 ---")
load_dotenv() # .envファイルがあれば読み込む

# 分割対象のサンプルテキスト (日本語・長文)
long_text_jp = """LangChain（ランクチェイン）は、大規模言語モデル（LLM）を活用したアプリケーション開発のための強力なフレームワークです。開発者は LangChain を使うことで、複雑な AI アプリケーションの構築プロセスを大幅に簡略化できます。

主な特徴として、様々なコンポーネント（プロンプトテンプレート、LLMラッパー、エージェント、メモリなど）をモジュール式に組み合わせられる点が挙げられます。これにより、特定のユースケースに合わせた柔軟なカスタマイズが可能になります。例えば、社内ドキュメントに基づいた質問応答システムの構築、複数のツールを自律的に使いこなすエージェントの作成、過去の対話履歴を記憶するチャットボットの開発などが実現できます。

LangChain は Python と TypeScript で提供されており、活発なオープンソースコミュニティによって支えられています。ドキュメントも充実しており、多くのサンプルコードが公開されているため、比較的容易に学習を始めることができます。しかし、提供される機能が多岐にわたるため、全体像を把握するにはある程度の学習時間が必要です。

特に重要なコンセプトの一つが LCEL (LangChain Expression Language) です。これは、パイプ演算子を使ってコンポーネントを直感的に連結し、処理フローを宣言的に記述するための言語（または記述方法）です。LCEL をマスターすることで、コードの可読性が向上し、より洗練されたアプリケーション開発が可能になります。

今後の AI アプリケーション開発において、LangChain のようなフレームワークの重要性はますます高まっていくと考えられます。最新の動向を追いかけ、基本的な使い方を習得しておくことは、AI エンジニアにとって有益でしょう。"""

print(f"--- サンプルテキスト準備完了 (文字数: {len(long_text_jp)}) ---")

# --- トークン数計算の準備 ---
# OpenAI モデルでよく使われる 'cl100k_base' エンコーディングを使用
# (注意: 使用する LLM や Embedding モデルに合わせて適切なエンコーダーを選ぶ必要があります)
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    print(f"--- tiktoken トークナイザ準備完了 (encoding: cl100k_base) ---")
    # 必要ならインストール: pip install tiktoken
except Exception as e:
    print(f"エラー: tiktoken の準備に失敗しました: {e}")
    print("   'pip install tiktoken' を実行してください。")
    sys.exit(1)

# テキストのトークン数を計算する関数を定義
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=() # 特殊トークンを許可しない (一般的なテキストカウント用)
    )
    return len(tokens)

# サンプルテキスト全体のトークン数も確認しておく (目安)
total_tokens = tiktoken_len(long_text_jp)
print(f"サンプルテキスト全体のトークン数 (目安): {total_tokens}")

# step13_text_splitters_revised.py (続き)

# Text Splitter を初期化 (トークン数基準、日本語対応)
text_splitter_token = RecursiveCharacterTextSplitter(
    chunk_size=200,         # 目標とする最大トークン数
    chunk_overlap=20,         # チャンク間の重複トークン数
    length_function=tiktoken_len, # トークン数を計算する関数を指定
    is_separator_regex=False,
    separators=["\n\n", "\n", "。", "、", " ", ""], # 日本語用の区切り文字リスト（デフォルトから少し調整も可能）
    # language="ja" # language 引数は langchain_text_splitters 0.2.0 以降で利用可能
                   # 利用可能な場合は、separators を自動設定してくれるため便利
)
print("\n--- RecursiveCharacterTextSplitter 初期化完了 (トークン数基準) ---")
print(f"  chunk_size (トークン数): {text_splitter_token._chunk_size}")
print(f"  chunk_overlap (トークン数): {text_splitter_token._chunk_overlap}")
# print(f"  language: {text_splitter_token._language}") # language 引数がある場合

# step13_text_splitters_revised.py (続き)
print("\n--- テキスト文字列の分割 (.split_text - トークン数基準) ---")

text_chunks_token = text_splitter_token.split_text(long_text_jp)
print(f"分割後のチャンク数: {len(text_chunks_token)}")

print("\n--- 分割されたチャンク (一部) ---")
for i, chunk in enumerate(text_chunks_token):
    chunk_token_count = tiktoken_len(chunk)
    print(f"--- チャンク {i+1} (トークン数: {chunk_token_count}, 文字数: {len(chunk)}) ---")
    if i < 3:
        print(chunk)
        # 重複部分の確認 (次のチャンクが存在する場合)
        if i + 1 < len(text_chunks_token):
            next_chunk = text_chunks_token[i+1]
            # 簡単な重複確認ロジック (厳密ではない)
            overlap_candidate = chunk[-text_splitter_token._chunk_overlap * 2 :] # 重複候補を少し広めに取る
            if next_chunk.startswith(overlap_candidate[-len(next_chunk):]): # 簡易的な前方一致
                 print("  (次のチャンクとの重複の可能性あり)")
    elif i == 3:
        print("...") # 4番目以降は省略
        break

# step13_text_splitters_revised.py (続き)
print("\n--- Document オブジェクトの分割 (.split_documents - トークン数基準) ---")

original_document_jp = Document(
    page_content=long_text_jp,
    metadata={"source": "jp_document.txt", "language": "ja"}
)
print(f"元の Document: (トークン数: {tiktoken_len(original_document_jp.page_content)}, metadata: {original_document_jp.metadata})")

# 同じ splitter を使って Document を分割
document_chunks_token = text_splitter_token.split_documents([original_document_jp])
print(f"分割後の Document 数: {len(document_chunks_token)}")

print("\n--- 分割された Document (最初のチャンク) ---")
if document_chunks_token:
    first_doc_chunk_token = document_chunks_token[0]
    chunk_token_count = tiktoken_len(first_doc_chunk_token.page_content)
    print(f"Type: {type(first_doc_chunk_token)}")
    print(f"Page Content (トークン数: {chunk_token_count}):")
    print(first_doc_chunk_token.page_content)
    print(f"Metadata: {first_doc_chunk_token.metadata}") # メタデータが引き継がれる
else:
    print("Document チャンクリストが空です。")

print("\n--- 処理終了 ---")
