"""
Out-expansion benchmarker
"""

from Logger import logging
from acronym_expander import RunConfig, AcronymOutExpanderFactory, TrainOutDataManager
from benchmarkers.benchmark_base import BenchmarkerBase
from benchmarkers.out_expansion.results_reporter import ResultsReporter
from text_preparation import sub_expansion_tokens_by_acronym


logger = logging.getLogger(__name__)


class Benchmarker(BenchmarkerBase):  # pylint: disable=too-few-public-methods
    """
    Out-expansion benchmarker
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        run_config: RunConfig,
        train_dataset_name,
        out_expander_name,
        out_expander_args,
        test_dataset_name=None,
        expansion_linkage=False,
        report_confidences=False,
        results_report=ResultsReporter,
    ):
        """
        :param run_config: data structure with run configuration parameters
        :param train_dataset_name: name of the train dataset
        :param out_expander_name: name of the out-expander technique to use
        :param out_expander_args: arguments for the out-expander technique
        :param test_dataset_name: name of the test dataset
        :param expansion_linkage: runs an expansion linkage process for the training data
        """
        if test_dataset_name is None:
            test_dataset_name = train_dataset_name

        self.expander_factory = AcronymOutExpanderFactory(
            out_expander_name, out_expander_args, run_config=run_config
        )

        experiment_parameters = {
            "out_expander": out_expander_name,
            "out_expander_args": out_expander_args,
        }

        super().__init__(
            run_config=run_config,
            experiment_parameters=experiment_parameters,
            train_dataset_name=train_dataset_name,
            test_dataset_name=test_dataset_name,
            results_report=results_report,
            expansion_linkage=expansion_linkage,
        )

        self.report_confidences = report_confidences

    def _create_expander(self, train_data_manager: TrainOutDataManager):
        return self.expander_factory.create_out_expander(train_data_manager)

    def _process_article(self, expander, article_for_testing, article_id, acronyms):
        return expander.process_article(article_for_testing, acronyms, article_id, expansion_confidences=self.report_confidences)
        
    def _preprocess_test_article(self, article, actual_expansions):
        new_actual_expansions = {}
        expansion_found = False
        processed_article = article
        sub_expansions = []
        for acronym, expansion in actual_expansions.items():

            processed_article, sub_count = sub_expansion_tokens_by_acronym(
                acronym, expansion, processed_article
            )
            if sub_count < 1:
                if expansion in sub_expansions:
                    logger.info(
                        "Skipped acronym %s, there is another acronym with same expansion %s in: %s",
                        acronym,
                        expansion,
                        article,
                    )
                else:
                    logger.error(
                        "Expansion %s for acronym %s not found in article %s",
                        expansion,
                        acronym,
                        article,
                    )
            else:
                new_actual_expansions[acronym] = expansion
                sub_expansions.append(expansion)
                expansion_found = True

        if not expansion_found:
            raise Exception("No expansion found in article %s" % article)

        return processed_article, new_actual_expansions
