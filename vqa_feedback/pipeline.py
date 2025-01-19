import torch
import numpy as np
from .vqa_models import ImageQAModel, build_prompt_func
from .wrench.labelmodel import *
from .questions_generation import generate_qid_data_from_prompt


label_models = {
    "MajorityWeightedVoting": MajorityWeightedVoting(),
    "MajorityVoting": MajorityVoting(),
    "DawidSknene": DawidSkene(),
    "MeTaL": MeTaL(),
    "EBCC": EBCC(), # need Weaklabel
    "FlyingSquid": FlyingSquid(),
    "GenerativeModel": GenerativeModel(),
    "GoldCondProb": GoldCondProb(), # need labels
    "NaiveBayesModel": NaiveBayesModel(),
    "Snorkel": Snorkel(),
    "Weapo": Weapo(), # cannot use
    "IBCC": IBCC(), # need Weaklabel
    "HyperLM": HyperLM() # need kernel_function
}


class BinaryQuesitonGenerator:
    def __init__(self, prompt):
        self.prompt = prompt

    def generate_binary_question(self):
        prompt = self.prompt
        self.questions = generate_qid_data_from_prompt(prompt)
        return self.questions


class VQAEnsembler:
    def __init__(self, vqa_models):
        self.vqa_models = vqa_models

    def generate_vqa_score_matrix(self, image, prompt):
        binaryQuesitonGenerator = BinaryQuesitonGenerator(prompt)
        questions = binaryQuesitonGenerator.generate_binary_question()
        qid2dependency = questions['qid2dependency']
        qid2questions = questions['qid2question']

        choice = ["Yes", "No"]
        score_matrix = []

        all_questions = []
        all_answers = []
        all_qa_results = []

        for model_idx, model in enumerate(self.vqa_models):
            vqa_scores = []
            model_questions = []
            model_answers = []
            model_qa_results = []

            for idx, question in qid2questions.items():
                answer = model.multiple_choice_qa(
                    image, question, choice,
                    prompt_func=build_prompt_func("Question: {question}\nselect from the following choices: {choices}")
                )
                binary_scores = 1 if answer['multiple_choice_answer'] == 'Yes' else 0
                vqa_scores.append(binary_scores)
                model_questions.append(question)
                model_answers.append(answer['multiple_choice_answer'])
                model_qa_results.append(binary_scores)

            for id, parent_ids in qid2dependency.items():
                any_parent_answered_no = False
                for parent_id in parent_ids:
                    if parent_id == 0:
                        continue
                    if vqa_scores[parent_id - 1] == 0:
                        any_parent_answered_no = True
                        break
                if any_parent_answered_no:
                    vqa_scores[int(id) - 1] = 0

            all_questions.append(model_questions)
            all_answers.append(model_answers)
            all_qa_results.append(model_qa_results)

            score_matrix.append(vqa_scores)

        score_matrix = np.array(score_matrix)

        return {
            "score_matrix": score_matrix,
            "all_questions": all_questions,
            "all_answers": all_answers,
            "all_qa_results": all_qa_results,
        }
        

class WeakSupervisor:
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.voting_strategy = None

    def _load_strategy(self):
        if self.voting_strategy is None:
            if self.strategy_name in label_models:
                self.voting_strategy = label_models[self.strategy_name]
            else:
                raise ValueError(f"Strategy '{self.strategy_name}' not found in the label_models dictionary")

    def compute_soft_score(self, score_matrix): 
        self._load_strategy()
        self.voting_strategy.fit(score_matrix.reshape(1, -1))
        soft_score = self.voting_strategy.predict_proba(score_matrix.reshape(1, -1))
        return soft_score


class Pipeline:
    def __init__(self, vqa_ensembler, weak_supervisor, logger=None): 
        self.vqa_ensembler = vqa_ensembler
        self.weak_supervisor = weak_supervisor
        self.logger = logger  # Controls whether to use the logger
    
    def process(self, image, prompt):
        # Generate the hard score matrix
        result = self.vqa_ensembler.generate_vqa_score_matrix(image, prompt)
        
        # Calculate soft score using the weak supervisor
        soft_score = self.weak_supervisor.compute_soft_score(result['score_matrix'])
        
        # The final score is the soft score at index 1
        final_score = soft_score[0][1]
        
        # If a logger is provided, log the results
        if self.logger:
            self.logger.log(
                image, prompt, result['all_questions'], result['all_answers'], result['all_qa_results'], final_score
            )
        
        # Return final_score
        return final_score

    def __call__(self, image, prompt):
        return self.process(image, prompt)
