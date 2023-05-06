from promptimize.prompt_cases import BasePromptCase, utils

# patched method while waiting for upstream PR to be merged

def patch_promptimize():
    def test(self):
        test_results = []
        for evaluator in self.evaluators:
            result = evaluator(self.response)
            if not (utils.is_numeric(result) and 0 <= result <= 1):
                raise Exception("Value should be between 0 and 1")
            test_results.append(result)

        if len(test_results):
            self.execution.score = sum(test_results) / len(test_results)
            self.execution.results = test_results
        self.was_tested = True


    BasePromptCase.test = test