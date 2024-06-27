from trilogy import Environment
from trilogy_nlp.enums import Provider
from trilogy_nlp.main import build_query
from trilogy_nlp.main_v2 import build_query as build_query_v2


class NLPEngine(object):
    def __init__(
        self,
        provider: Provider,
        model: str | None = None,
        api_key: str | None = None,
        args: dict = {},
    ):
        self.provider = provider
        self.debug = False
        self.model = model
        self.api_key = api_key
        self.llm = self.create_llm()

    def create_llm(self):
        if self.provider == Provider.OPENAI:
            from langchain_community.chat_models import ChatOpenAI

            llm = ChatOpenAI(
                model_name=self.model if self.model else "gpt-3.5-turbo",
                openai_api_key=self.api_key,
                # model_kwargs=chat_openai_model_kwargs,
            )
        elif self.provider == Provider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=self.model if self.model else "gemini-pro",
                convert_system_message_to_human=True,
                google_api_key=self.api_key,
            )
        else:
            raise NotImplementedError(f"Unsupported provider {self.provider}")
        return llm

    def test_connection(self):
        local = self.create_llm()
        result = local.invoke("Hello")
        print(result.content)

    def build_query_from_text(
        self, input_text: str, env: Environment, version: int = 1
    ):
        if version == 2:
            return build_query_v2(
                input_text=input_text,
                input_environment=env,
                debug=self.debug,
                llm=self.llm,
            )
        return build_query(
            input_text=input_text,
            input_environment=env,
            debug=self.debug,
            llm=self.llm,
        )
