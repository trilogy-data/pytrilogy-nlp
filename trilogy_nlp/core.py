from langchain.globals import set_llm_cache
from trilogy import Environment, Executor
from trilogy.core.models import ProcessedQuery
from trilogy.executor import CursorResult

from trilogy_nlp.enums import CacheType, Provider
from trilogy_nlp.instrumentation import EventTracker
from trilogy_nlp.main import build_query

DEFAULT_GPT = "gpt-4o-mini"
DEFAULT_GEMINI = "gemini-pro"

DEFAULT_MAX_TOXENS = 6500

# 0 to 1.0, scaled by provider
DEFAULT_TEMPERATURE = 0.25


class NLPEngine(object):
    def __init__(
        self,
        provider: Provider,
        model: str | None = None,
        api_key: str | None = None,
        cache: CacheType | None = None,
        cache_kwargs: dict | None = None,
        instrumentation: EventTracker | None = None,
    ):
        self.provider = provider
        self.debug = False
        self.model = model
        self.api_key = api_key
        self.llm = self.create_llm()
        self.cache = self.create_cache(cache, cache_kwargs or {}) if cache else None
        self.instrumentation = instrumentation

    def create_cache(self, cache: CacheType, cache_kwargs: dict):
        from langchain_core.caches import BaseCache

        cache_instance: BaseCache
        if cache == CacheType.SQLLITE:
            from langchain_community.cache import SQLiteCache

            cache_instance = SQLiteCache(**cache_kwargs)
        elif cache == CacheType.MEMORY:
            from langchain_community.cache import InMemoryCache

            cache_instance = InMemoryCache()
        else:
            return None
        set_llm_cache(cache_instance)
        return cache_instance

    def create_llm(self):
        if self.provider == Provider.OPENAI:
            import openai
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model_name=self.model if self.model else DEFAULT_GPT,
                openai_api_key=self.api_key,
                # openai temperature is 0 to 2.0
                temperature=DEFAULT_TEMPERATURE * 2,
                # model_kwargs=chat_openai_model_kwargs,
            ).with_retry(
                retry_if_exception_type=(
                    openai.APIConnectionError,
                    openai.APIError,
                    openai.APITimeoutError,
                )
            )
        elif self.provider == Provider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=self.model if self.model else DEFAULT_GEMINI,
                convert_system_message_to_human=True,
                google_api_key=self.api_key,
                # openai temperature is 0 to 2.0
                temperature=DEFAULT_TEMPERATURE * 2,
            ).with_retry()
        elif self.provider == Provider.LLAMAFILE:
            from langchain_community.chat_models import ChatOpenAI

            llm = ChatOpenAI(
                model_name=self.model if self.model else "not-applicalbe",
                openai_api_key="not-required",
                base_url="http://localhost:8080/v1",
                temperature=DEFAULT_TEMPERATURE,
                # model_kwargs=chat_openai_model_kwargs,
            ).with_retry()
        else:
            raise NotImplementedError(f"Unsupported provider {self.provider}")
        return llm

    def test_connection(self):
        local = self.create_llm()
        result = local.invoke("Hello")
        print(result.content)

    def generate_query(self, text: str, env: Environment | Executor) -> ProcessedQuery:
        if isinstance(env, Executor):
            env = env.environment
        # avoid mutating our model
        env = env.model_copy(deep=True)
        return build_query(
            input_text=text,
            input_environment=env,
            debug=self.debug,
            llm=self.llm,
        )

    def run_query(self, text: str, executor: Executor) -> CursorResult:
        query = self.generate_query(text, executor)
        return executor.execute_query(query)
