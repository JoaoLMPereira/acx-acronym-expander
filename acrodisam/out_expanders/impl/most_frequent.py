from collections import Counter

from inputters import TrainOutDataManager

from .._base import OutExpanderFactory, OutExpanderArticleInput, OutExpander


class FactoryMostFrequent(OutExpanderFactory):
    def __init__(self, *args, **kwargs):
        pass

    def get_expander(
        self, train_data_manager: TrainOutDataManager, execution_time_observer=None
    ):
        # selects most frequent expansion per acronym
        acronym_expansion = {}
        execution_time_observer.start()
        for acronym, expansion_articles in train_data_manager.acronym_db.items():
            counter = Counter([exp_article[0] for exp_article in expansion_articles])
            if len(counter) > 0:
                most_freq = counter.most_common(1)[0]
                expansion = most_freq[0]
                confidence = most_freq[1] / len(expansion_articles)

                acronym_expansion[acronym] = (expansion, confidence)

        execution_time_observer.stop()
        return _ExpanderMostFrequent(acronym_expansion)


class _ExpanderMostFrequent(OutExpander):
    def __init__(self, acronym_expansion: dict):
        self.acronym_expansion = acronym_expansion

    def process_article(self, out_expander_input: OutExpanderArticleInput):
        predicted_expansions = []
        for acronym in out_expander_input.acronyms_list:
            predicted_expansions.append(self.acronym_expansion[acronym])

        return predicted_expansions
