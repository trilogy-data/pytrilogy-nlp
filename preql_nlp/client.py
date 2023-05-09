from preql import Environment, Executor
from preql_nlp.main import build_query, answer_is_reasonable
from dataclasses import dataclass

from time import sleep
from typing import List

@dataclass
class NlpPreqlModelClient:

    openai_model: str
    preql_model: Environment
    preql_executor: Executor

    def answer(self, question: str) -> List[tuple]:
        max_retries = 3
        retries = 0
        while retries < max_retries:
            query = build_query(question, self.preql_model, debug=False, log_info=True, model=self.openai_model)
            results = self.preql_executor.execute_query(query)
            cols = results.keys()

            res = []
            for r in results:
                res.append(r)
            
            if self.answer_is_reasonable(question, res, cols):
                return res
            else:
                retries += 1
                sleep(1)
        
        raise Exception(f"Answer not reasonable after {max_retries} retries")
    
    def answer_is_reasonable(self, question, results, columns) -> bool:
        return answer_is_reasonable(question, results, columns)

