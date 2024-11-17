import os
import nest_asyncio  # noqa: E402
nest_asyncio.apply()

# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

##### LLAMAPARSE #####
from llama_parse import LlamaParse

import pickle

llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

class PdfLoader:
    def __init__(self, document_path, instruction=None) -> None:
        self.document_path = document_path
        if instruction is not None:
            self.instruction = instruction
        else:
            self.instruction = """提供されたドキュメントは、リーダー電子による技術情報シリーズ「Vol.02 HDR 測定」です。
        このドキュメントは、HDR（ハイダイナミックレンジ）技術およびその測定に関連するトピックを取り扱っています。
        HDRの概念、SDRとの比較、HDR測定技術に関する詳細な説明が含まれており、図表やテーブルを使用して、技術的な内容が説明されています。
        本ドキュメントには、多くの図、テーブル、技術的な用語が含まれており、理解を深めるために重要です。

        各章の内容は以下の通りです：
        01 HDRとは
        02 ガンマとは
        03 SDR方式
        04 HDR方式
        05 HDRの測定
        06 用語集

        質問に答える際は、特に図表やテーブル、定義に関して正確であることを心掛けてください。"""

    def load(self):
        if os.path.exists(self.document_path):
            with open(self.document_path, "rb") as f:
                parsed_data = pickle.load(f)
        else:
            parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown", parsing_instruction=self.instruction, languagla="ja")
            parsed_data = parser.load_data(self.document_path)

            # Save the parsed data to a file
            with open(self.document_path, "wb") as f:
                pickle.dump(parsed_data, f)

            # Set the parsed data to the variable
        return parsed_data
