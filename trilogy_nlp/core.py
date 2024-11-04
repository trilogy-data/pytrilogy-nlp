from trilogy import Environment
from trilogy_nlp.enums import Provider, CacheType
from trilogy_nlp.main import build_query
from trilogy_nlp.main_v2 import build_query as build_query_v2
from langchain.globals import set_llm_cache

DEFAULT_GPT = "gpt-3.5-turbo"
DEFAULT_GEMINI = "gemini-pro"

DEFAULT_MAX_TOXENS = 6500


class NLPEngine(object):
    def __init__(
        self,
        provider: Provider,
        model: str | None = None,
        api_key: str | None = None,
        cache: CacheType | None = None,
        cache_kwargs: dict | None = None,
    ):
        self.provider = provider
        self.debug = False
        self.model = model
        self.api_key = api_key
        self.llm = self.create_llm()
        self.cache = self.create_cache(cache, cache_kwargs or {})

    def create_cache(self, cache: CacheType, cache_kwargs: dict):
        if cache == CacheType.SQLLITE:
            from langchain.cache import SQLiteCache

            cache = SQLiteCache(**cache_kwargs)
        elif cache == CacheType.MEMORY:
            from langchain.cache import InMemoryCache

            cache = InMemoryCache()
        else:
            return None
        set_llm_cache(cache)
        return cache

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
        elif self.provider == Provider.LLAMAFILE:
            from langchain_community.chat_models import ChatOpenAI

            llm = ChatOpenAI(
                model_name=self.model if self.model else "gpt-3.5-turbo",
                openai_api_key="not-required",
                base_url="http://localhost:8080/v1",
                # model_kwargs=chat_openai_model_kwargs,
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
