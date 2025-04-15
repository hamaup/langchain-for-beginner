# step24_custom_tool_pydantic_final.py
import os
import sys
from dotenv import load_dotenv
from typing import List, Optional # 型ヒントで使用
import json # スキーマ表示用

# @tool デコレータをインポート
try:
    from langchain_core.tools import tool
    print("`@tool` デコレータをインポートしました (from langchain_core.tools)。")
except ImportError:
    print("エラー: `@tool` デコレータが見つかりません。")
    print("   'pip install langchain-core' を確認してください。")
    sys.exit(1)

# Pydantic をインポート (v2 を想定)
try:
    from pydantic import BaseModel, Field
    print("Pydantic (v2) をインポートしました。")
    # Pydantic v1 を使いたい場合は from pydantic.v1 import ...
except ImportError:
    print("エラー: Pydantic が見つかりません。")
    print("   'pip install pydantic' を実行してください。")
    sys.exit(1)

print("--- 必要なモジュールのインポート完了 ---")

load_dotenv()
print("環境変数をロードしました (必要に応じて)")
# step24_custom_tool_pydantic_final.py (続き)

# --- 例: ユーザー情報を検索するツールの入力スキーマ ---
class UserSearchInput(BaseModel):
    user_id: str = Field(description="情報を検索したいユーザーの一意なID。")
    include_details: bool = Field(default=False, description="Trueの場合、注文履歴などの詳細情報も含める。デフォルトはFalse。")

print("\n--- ユーザー検索ツールの入力スキーマ (UserSearchInput) 定義完了 ---")
# step24_custom_tool_pydantic_final.py (続き)

# --- ユーザー情報検索ツールの実装 ---
@tool(args_schema=UserSearchInput, description="指定されたユーザーIDに基づいてユーザー情報を検索し、基本情報または詳細情報（注文履歴など）を返すために使用します。")
def search_user_info(user_id: str, include_details: bool = False) -> dict:
    """ユーザーIDで情報を検索し、辞書形式で結果を返します。""" # descriptionがあれば補助的
    print(f"ツール実行: search_user_info(user_id='{user_id}', include_details={include_details})")
    # --- ここに実際のデータベース検索などの処理が入る ---
    # (今回はダミーデータを返す)
    user_db = {
        "user123": {"name": "田中 太郎", "email": "tanaka@example.com", "orders": ["orderA", "orderB"]},
        "user456": {"name": "佐藤 花子", "email": "sato@example.com", "orders": ["orderC"]},
    }
    if user_id in user_db:
        user_data = user_db[user_id].copy() # 元データを変更しないようにコピー
        if include_details:
            # 詳細情報を含める場合は全データを返す
            return user_data
        else:
            # 基本情報だけ返す
            user_data.pop("orders", None) # 注文履歴キーを削除 (なければ何もしない)
            return user_data
    else:
        return {"error": f"ユーザーID '{user_id}' が見つかりません。"}

print("--- ユーザー検索ツール (search_user_info) 定義完了 ---")

# --- (参考) シンプルな型ヒントと Docstring で推測させる場合 ---
@tool
def multiply_simple(a: int, b: int) -> int:
    """二つの整数 a と b の積を計算します。掛け算が必要な時に使ってください。
    Args:
        a (int): かけられる数。
        b (int): かける数。
    """
    print(f"ツール実行: multiply_simple({a}, {b})")
    return a * b

print("--- (参考) シンプルな掛け算ツール (multiply_simple) 定義完了 ---")
# step24_custom_tool_pydantic_final.py (続き)

print("\n--- 作成されたツールの確認 ---")

# search_user_info ツール
print(f"\n[search_user_info ツール]")
print(f"型: {type(search_user_info)}")
print(f"名前: {search_user_info.name}")
print(f"説明: {search_user_info.description}")
print("引数スキーマ (Pydantic V2):")
# Pydantic V2 の場合 .schema() ではなく .model_json_schema() が推奨されることも
try:
    print(json.dumps(search_user_info.args_schema.model_json_schema(), indent=2, ensure_ascii=False))
except AttributeError: # 古い Pydantic や LangChain の場合 .schema()
    try:
        print(json.dumps(search_user_info.args_schema.schema(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"スキーマ表示エラー: {e}")
        print(f"args_schema: {search_user_info.args_schema}")


# multiply_simple ツール (参考)
print(f"\n[multiply_simple ツール (参考)]")
print(f"型: {type(multiply_simple)}")
print(f"名前: {multiply_simple.name}")
print(f"説明: {multiply_simple.description}")
print("引数スキーマ (Docstring/Type Hint 推論):")
# こちらは Pydantic モデルではないので .args を使うのが一般的
# (ただし .args も将来非推奨になる可能性はあります)
print(json.dumps(multiply_simple.args, indent=2, ensure_ascii=False))


# ツールを直接実行してみる
print("\n--- ツールを直接実行してみる ---")
try:
    user_info = search_user_info.invoke({"user_id": "user123", "include_details": True})
    print(f"search_user_info('user123', True) の結果: {user_info}")

    user_info_simple = search_user_info.invoke({"user_id": "user456"}) # include_details はデフォルト (False)
    print(f"search_user_info('user456') の結果: {user_info_simple}")

    multiply_result = multiply_simple.invoke({"a": 12, "b": 5})
    print(f"multiply_simple(12, 5) の結果: {multiply_result}")

except Exception as e:
    print(f"エラー: ツールの直接実行中にエラーが発生しました: {e}")


print("\n--- 処理終了 ---")