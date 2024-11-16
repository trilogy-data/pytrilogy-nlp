from datetime import datetime


def get_wiki_tool():
    from langchain_community.tools.wikidata.tool import (
        WikidataAPIWrapper,
        WikidataQueryRun,
    )

    wiki = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())  # type: ignore
    wiki.description = (
        "Look up information on a specific string from Wikipedia. Use to get context"
    )
    return wiki


def get_today_date(query: str) -> str:
    """
    Useful to get the date of today.
    """
    # Getting today's date in string format
    today_date_string = datetime.now().strftime("%Y-%m-%d")
    return today_date_string
