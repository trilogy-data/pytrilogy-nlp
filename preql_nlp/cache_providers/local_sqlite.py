from preql_nlp.cache_providers.base import BaseCache
import sqlite3

DEFAULT_SQLITE_ADDRESS = "local_prompt_cache.db"



class SqlliteCache(BaseCache):


    def __init__(self, sqlite_address: str = DEFAULT_SQLITE_ADDRESS):
        self.sqlite_address = sqlite_address

    
    def retrieve(self, prompt_hash: str) -> str | None:
        print("checking for cache with prompt hash ", prompt_hash)
        con = sqlite3.connect(self.sqlite_address)
        cur = con.cursor()
        cur.execute(
            "create table if not exists prompt_cache (cache_id string, prompt_type string, response string)"
        )
        res = cur.execute(
            "select response, prompt_type from prompt_cache where cache_id = ?",
            (prompt_hash,),
        )
        current = res.fetchone()
        if current:
            print("got cached response of type ", current[1])
            return current[0]
        return None



    def stash(self, prompt_hash: str, category: str, result: str):
        con = sqlite3.connect(self.sqlite_address)
        cur = con.cursor()
        cur.execute(
            "create table if not exists prompt_cache (cache_id string, prompt_type string, response string)"
        )
        cur.execute(
            "insert into prompt_cache select ?, ?,  ?", (prompt_hash, category, result)
        )
        con.commit()