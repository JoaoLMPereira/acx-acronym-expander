import random

from inputters import TrainOutDataManager

from .._base import OutExpanderFactory, OutExpanderArticleInput, OutExpander


class FactoryRandom(OutExpanderFactory):
    def __init__(self, *args, **kwargs):
        pass

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        acronym_expansions = {}
        execution_time_observer.start()
        for acronym, expansion_articles in train_data_manager.acronym_db.items():
            distinct_expansions = list(
                {exp_article[0] for exp_article in expansion_articles}
            )
            acronym_expansions[acronym] = distinct_expansions

        execution_time_observer.stop()
        return _ExpanderRandom(acronym_expansions)


class _ExpanderRandom(OutExpander):
    def __init__(self, acronym_expansion: dict):
        self.acronym_expansion = acronym_expansion

    def process_article(self, out_expander_input: OutExpanderArticleInput):
        predicted_expansions = []
        for acronym in out_expander_input.acronyms_list:
            expansions = self.acronym_expansion[acronym]
            random_expansion = random.choice(expansions)
            confidence = 1 / len(expansions)
            predicted_expansions.append((random_expansion, confidence))

        return predicted_expansions
