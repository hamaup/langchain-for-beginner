# step12_document_loaders_revised.py
import os
import sys
from dotenv import load_dotenv

# Document Loaders を langchain_community からインポート
try:
    # 現在推奨されるインポートパス (2025年4月時点)
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    print("TextLoader と PyPDFLoader をインポートしました (from langchain_community)。")
except ImportError:
    print("エラー: langchain-community が見つかりません。")
    print("   'pip install langchain-community' を実行してください。")
    sys.exit(1)

# Document オブジェクトの型を確認するためにインポート (任意)
from langchain_core.documents import Document

print("--- 必要なモジュールのインポート完了 ---")
load_dotenv() # .envファイルがあれば読み込む

# step12_document_loaders_revised.py (続き)
print("\n--- テキストファイル (sample.txt) の読み込み ---")

txt_file_path = "sample.txt"

if not os.path.exists(txt_file_path):
    print(f"エラー: {txt_file_path} が見つかりません。")
else:
    try:
        # TextLoader を初期化。日本語を含む場合は encoding='utf-8' を指定。
        # 指定しない場合のデフォルトエンコーディングは実行環境に依存します。
        text_loader = TextLoader(txt_file_path, encoding='utf-8')
        print(f"TextLoader の準備完了: {txt_file_path}")

        # .load() で Document オブジェクトのリストを取得
        text_documents = text_loader.load()
        print(f"読み込み完了。{len(text_documents)} 個の Document オブジェクトを取得しました。")

        # load() の結果を確認 (通常は1つだが、実装によっては複数になる可能性もある)
        if text_documents:
            print("\n--- 読み込んだ Document の内容 (Text) ---")
            for i, doc in enumerate(text_documents):
                print(f"  --- Document {i+1} ---")
                print(f"  Type: {type(doc)}")
                print(f"  Page Content (最初の50文字): {doc.page_content[:50]}...")
                print(f"  Metadata: {doc.metadata}") # 'source' が含まれるはず
        else:
            print("Document オブジェクトが空です。")

    except Exception as e:
        print(f"テキストファイルの読み込み中にエラーが発生しました: {e}")

# step12_document_loaders_revised.py (続き)
print("\n--- PDF ファイル (sample.pdf) の読み込み ---")

pdf_file_path = "sample.pdf"

if not os.path.exists(pdf_file_path):
    print(f"エラー: {pdf_file_path} が見つかりません。")
else:
    try:
        pdf_loader = PyPDFLoader(pdf_file_path)
        print(f"PyPDFLoader の準備完了: {pdf_file_path}")

        # .load() で Document オブジェクトのリストを取得 (通常、ページごとに分割)
        pdf_documents = pdf_loader.load()
        print(f"読み込み完了。{len(pdf_documents)} 個の Document オブジェクトを取得しました。")

        # 読み込んだ内容を確認 (最初のページの Document だけ表示)
        if pdf_documents:
            print("\n--- 読み込んだ Document の内容 (PDF - 1ページ目) ---")
            first_page_doc = pdf_documents[0]
            print(f"Type: {type(first_page_doc)}")
            print(f"Page Content (最初の50文字): {first_page_doc.page_content[:50]}...")
            print(f"Metadata: {first_page_doc.metadata}") # 'source' と 'page' が含まれるはず

            # (参考) 2ページ目以降も確認する場合
            if len(pdf_documents) > 1:
                 second_page_doc = pdf_documents[1]
                 print("\n--- 読み込んだ Document の内容 (PDF - 2ページ目) ---")
                 print(f"Metadata: {second_page_doc.metadata}") # ページ番号が 1 になっているはず
                 print(f"Page Content (最初の50文字): {second_page_doc.page_content[:50]}...")
            print("\n注意: PDFからのテキスト抽出精度は、PDFの構造（テキストベースか画像か、レイアウトの複雑さなど）に大きく依存します。")
        else:
            print("Document オブジェクトが空です。")

    except ImportError:
        print("エラー: PyPDFLoader を使用するには 'pypdf' が必要です。")
        print("   'pip install pypdf' を実行してください。")
    except Exception as e:
        print(f"PDF ファイルの読み込み中にエラーが発生しました: {e}")
        print("   考えられる原因: PDFファイルが破損している、パスワード保護、複雑なレイアウト、pypdf未対応形式など。")

print("\n--- 処理終了 ---")
