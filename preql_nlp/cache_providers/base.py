



class BaseCache:

    def store(self, prompt_hash, category, result):
        raise NotImplementedError()
    
    def retrieve(self, prompt_hash):
        raise NotImplementedError()