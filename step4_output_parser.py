# step4_output_parser_meeting.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# プロンプトテンプレートをインポート
from langchain.prompts import ChatPromptTemplate
# Output Parser関連をインポート
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# 環境変数の読み込み
load_dotenv()
print("--- 環境変数読み込み完了 ---")

# LLMの準備 (temperature=0 で結果を安定させる)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print(f"--- LLM準備完了: {llm.model_name} (temperature={llm.temperature}) ---")

# 解析対象の会議議事録テキスト (日本語)
meeting_minutes_text = """
# 定例プロジェクト会議 議事録

**日付:** 2025年4月7日
**場所:** 第3会議室
**参加者:** 山田太郎、佐藤花子、田中一郎、鈴木健太（一部リモート）

## 議題
1. 新機能Aの開発進捗確認
2. プロモーション計画について
3. 次回スケジュール

## 議論内容・決定事項
- **新機能A:**
    - 山田より進捗報告。実装は80%完了。残りUI調整とテスト。
    - 佐藤より、追加仕様の提案があったが、今回は見送り、次期フェーズで検討することに決定。
    - UI調整は田中が担当。テストは鈴木が担当。
    - **決定:** 今週末までにUI調整を完了し、来週月曜日からテスト開始。
- **プロモーション計画:**
    - 佐藤より計画案の説明。Web広告とSNSキャンペーンを軸とする。
    - 田中より、ターゲット層への訴求力について意見。もう少し具体的なユースケースを示すべきでは？
    - **決定:** 佐藤は田中からのフィードバックを反映し、計画案を修正。水曜日までに再提出。
- **次回スケジュール:**
    - **決定:** 次回定例会議は来週月曜日の10:00から同会議室にて開催。

## 次回までのアクション
- 山田: 特になし（UI調整、テストのサポート）
- 佐藤: プロモーション計画の修正・再提出（水曜〆切）
- 田中: 新機能AのUI調整（今週末〆切）
- 鈴木: 新機能Aのテスト準備（来週月曜開始）

## その他
- 特になし

以上
"""

# 1. 出力形式の設計図 (ResponseSchema) を作る
purpose_schema = ResponseSchema(name="purpose",
                                description="この会議の主な目的や議題を簡潔な文字列で記述してください。")
decisions_schema = ResponseSchema(name="decisions",
                                  description="会議で決定された事項を抽出し、文字列のPythonリスト形式で出力してください。")
next_actions_schema = ResponseSchema(name="next_actions",
                                    description="次回会議までに行うべき具体的なアクション（担当者含む）を抽出し、文字列のPythonリスト形式で出力してください。")
attendees_schema = ResponseSchema(name="attendees",
                                 description="会議の参加者名を抽出し、文字列のPythonリスト形式で出力してください。")
response_schemas = [purpose_schema, decisions_schema, next_actions_schema, attendees_schema]

# 2. 解析器 (StructuredOutputParser) を作る
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 3. AIへの出力形式指示 (format_instructions) を取得
format_instructions = output_parser.get_format_instructions()
print("--- 生成されたフォーマット指示 (AIへの依頼内容) ---")
print(format_instructions)

# 4. プロンプトテンプレートを作成 (指示を埋め込む)
template_str = """\
以下の会議議事録テキストを分析し、指定された情報を抽出してください。

会議議事録テキスト:
{minutes}

抽出情報と形式:
{format_instructions}""" # <<< フォーマット指示を末尾に埋め込む

prompt = ChatPromptTemplate.from_template(template=template_str)

# 5. プロンプトに変数を埋め込む
messages = prompt.format_messages(minutes=meeting_minutes_text,
                                 format_instructions=format_instructions)

print("\n--- LLMへの最終的な入力メッセージ ---")
# print(messages[0].content) # 必要ならコメント解除して確認

# 6. LLMを呼び出し
response = llm.invoke(messages)
print("\n--- LLMからの応答 (文字列) ---")
print(response.content)

# 7. 応答文字列をパースしてPython辞書に変換
try:
    output_dict = output_parser.parse(response.content)
    print("\n--- パース後のPython辞書 ---")
    print(output_dict)
    print("パース後の型:", type(output_dict))

    # 辞書から値を取得
    print("\n会議の目的:", output_dict.get('purpose'))
    print("決定事項:", output_dict.get('decisions'))
    print("次回のアクション:", output_dict.get('next_actions'))
    print("参加者:", output_dict.get('attendees'))

except Exception as e:
    print("\n--- パースエラー ---")
    print(f"エラー内容: {e}")
    print("考えられる原因: LLMの応答が期待されたJSON形式でない可能性があります。")
    print("LLMの応答内容を確認してください:")
    print(response.content)

print("\n--- 処理終了 ---")