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
