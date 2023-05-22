



class BaseCache:

    def store(self, prompt_hash: str, category: str, result: str):
        raise NotImplementedError()
    
    def retrieve(self, prompt_hash:str):
        raise NotImplementedError()