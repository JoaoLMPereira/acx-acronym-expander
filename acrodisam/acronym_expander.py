"""
Acronym expanders that receive document text and outputs acronym-expansion pairs
"""
import collections
import re
import json
from typing import Callable, Dict

from AcroExpExtractors.AcroExpExtractor_Original_Schwartz_Hearst import (
    AcroExpExtractor_Original_Schwartz_Hearst,
)
from AcroExpExtractors.AcroExpExtractor_Schwartz_Hearst import (
    AcroExpExtractor_Schwartz_Hearst,
)
from AcroExpExtractors.AcroExpExtractor_Yet_Another_Improvement2 import (
    AcroExpExtractor_Yet_Another_Improvement,
)
from AcroExpExtractors.AcroExpExtractor_AAAI_Schwartz_Hearst import (
    AcroExpExtractor_AAAI_Schwartz_Hearst,
)

from DatasetParsers.FullWikipedia import text_preprocessing
from expansion_module import ExpansionModuleEnum
from helper import AcronymExpansion, ExecutionTimeObserver, get_args_to_pass
from inputters import InputArticle, TrainOutDataManager, TrainInDataManager
from links_follower import LinksFollower
from Logger import logging
from out_expanders import (
    OutExpander,
    OutExpanderArticleInput,
    OutExpanderFactory,
    get_out_expander_factory,
)
from run_config import RunConfig
from string_constants import min_confidence

logger = logging.getLogger(__name__)


class AcronymOutExpander:  # pylint: disable=too-few-public-methods
    """
    Class to out-expand the acronyms in text
    """

    def __init__(
        self,
        expander: OutExpander,
        acronym_db: Dict,
        bypass_db: bool = False,
    ):
        """
        Args:
            expander: OutExpander to use
            acronym_db: DataCreators.AcronymDB
        """

        self.acronym_db = acronym_db

        self.acronym_expander = expander
        
        self.bypass_db = bypass_db

    def _get_train_instances(self, acronym):
        expansions_per_article = self.acronym_db.get(acronym, [])

        # create training data
        x_train, y_train = [], []
        for expansion, article_id in expansions_per_article:
            x_train.append(article_id)
            y_train.append(expansion)
        return x_train, y_train

    def _process_expander_inputs(
        self, article, acronyms, test_article_id, expansions_per_acronym
    ) -> OutExpanderArticleInput:
        # expander_inputs = []
        expander_input = OutExpanderArticleInput(
            test_article_id=test_article_id, article=article
        )
        for acronym in acronyms:
            if not self.bypass_db:
                x_train, y_train = self._get_train_instances(acronym)
    
                options = set(y_train)
                
                if len(x_train) == 0:
                    # no point using prediction, no training data
                    # move to next expander
                    continue
                if len(options) == 1:
                    # no point using prediction, all same class
                    # predict as the only present class
                    expansion = AcronymExpansion(
                        expansion=y_train[0],
                        # expander=expander.getType(),
                        confidence=min_confidence,
                        options=options,
                    )
                    expansions_per_acronym[acronym] = expansion
                    continue
                
                expander_input.add_acronym(
                    acronym=acronym,
                    train_intances_ids=x_train,
                    train_instance_expansions=y_train,
                    distinct_expansions=options,
                )
            else:
                expander_input.add_acronym(
                    acronym=acronym,
                    train_intances_ids=[],
                    train_instance_expansions=set(),
                    distinct_expansions=[],
                )      
        if len(expander_input.acronyms_list) < 1:
            return None
        return expander_input

    def process_article(
        self,
        article: InputArticle,
        acronyms,
        test_article_id=None,
        expansion_confidences=False,
    ) -> dict:
        """
        takes an article and the existing acronyms without expansion in text and returns expansions
        for those acronyms

        Args:
            article: InputArticle provides the text either plain or preprocessed
            acronyms: list of acronyms with no expansion in text
            test_article_id: article identifier
            rank_expansions: boolean to return a ranked list of expansions per acronym
            instead of the most likely expansion
        Returns:
            dict: dictionary of acronym:out expansion found
        """
        expansions_per_acronym = {}
        expander_input = self._process_expander_inputs(
            article, acronyms, test_article_id, expansions_per_acronym
        )
        if not expander_input:
            return {}

        if expansion_confidences:
            expander_outputs = self.acronym_expander.process_article_return_confidences(
                expander_input)

        else:
            expander_outputs = self.acronym_expander.process_article(expander_input)

        for acronym, options, exp_output in zip(
            expander_input.acronyms_list,
            expander_input.distinct_expansions_list,
            expander_outputs,
        ):
            if not expansion_confidences:
                expansion = exp_output[0]
                confidence = exp_output[1]
                if expansion:
                    expansions_per_acronym[acronym] = AcronymExpansion(
                        expansion=expansion,
                        confidence=confidence,
                        options=options,
                    )
            else:
                exp, conf = max(exp_output.items(), key=lambda item: item[1])
                expansions_per_acronym[acronym] = AcronymExpansion(
                    expansion=exp,
                    confidence=conf,
                    options=exp_output
                )

        return expansions_per_acronym

    

class AcronymOutExpanderFactory:  # pylint: disable=too-few-public-methods
    """
    Acronym Out Expander Factory
    """

    def __init__(
        self,
        out_expander_name: str,
        out_expander_args,
        run_config: RunConfig,
    ):
        """
        Args:
            out_expander_name: Out expander technique name
            out_expander_args: Arguments for the out expander
            run_config: general run configurations
        """

        args, kwargs = get_args_to_pass(out_expander_args)
        self.expander_factory = get_out_expander_factory(
            out_expander_name, *args, run_config=run_config, **kwargs
        )

    def create_out_expander(
        self, train_data_manager: TrainOutDataManager, bypass_db:bool = False
    ) -> AcronymOutExpander:
        """
        Args:
            train_data_manager: manager of the data to create the out expander
        Returns:
            Acronym Out Expander trained on the data provided by the train_data_manager
        """
        model_exec_time = ExecutionTimeObserver()
        if isinstance(self.expander_factory, OutExpanderFactory):
            out_expander = self.expander_factory.get_expander(
                train_data_manager=train_data_manager,
                execution_time_observer=model_exec_time,
            )
        else:
            out_expander = self.expander_factory
        return (
            AcronymOutExpander(
                expander=out_expander, acronym_db=train_data_manager.get_acronym_db(), bypass_db=bypass_db
            ),
            model_exec_time,
        )


class AcronymExpander:  # pylint: disable=too-few-public-methods
    """
    Class to process plain text that may contain html tag links and expand the acronyms in it
    """

    def __init__(
        self,
        text_preprocessor,
        in_expander,
        acronym_out_expander: AcronymOutExpander,
        links_follower=None,
    ):
        """
        Args:
        text_preprocessor: text preprocessor function
        in_expander: in expander method to use
        acronym_out_expander: AcronymOutExpander object contains out expander method to use,
        links_follower: follow links object to process html tags and extract acronyms from
         the pages pointed by those links
        """

        self.text_preprocessor = text_preprocessor
        self.in_expander = in_expander

        self.out_expander = acronym_out_expander

        self.links_follower = links_follower

    def process_article(
        self,
        article: InputArticle,
        test_article_id=None,
        text_with_links=None,
        base_url=None,
    ):
        """
        takes text and returns the expanded acronyms in it

        Args:
        article: InputArticle provides the text either plain or preprocessed
        test_article_id: article identifier
        text_with_links: text with html tags if applicable
        base_url: test article base url for html tag manipulation

        Returns:
        dictionary of acronym:[expansion, expansion_method] found
        """
        # soup = BeautifulSoup(text, 'lxml')
        raw_text = article.get_raw_text()
        expansion_dict = self.in_expander.get_all_acronym_expansion(text=raw_text)
        # acronyms = [acronym for acronym, expansion in expansionDict.items() if not expansion]
        links_followed = [] # just for tracking purposes

        expansion_dict = {
            k: ([v, ExpansionModuleEnum.in_expander] if isinstance(v, str) else v)
            for k, v in expansion_dict.items()
        }

        if self.links_follower and text_with_links:
            expansion_dict, links_followed = self.links_follower(
                text_with_links, expansion_dict, base_url
            )

            expansion_dict = {
                k: ([v, ExpansionModuleEnum.link_follower] if isinstance(v, str) else v)
                for k, v in expansion_dict.items()
            }

        acronyms_without_expansions = [
            acronym
            for acronym, expansion in expansion_dict.items()
            if expansion is None
        ]

        if len(acronyms_without_expansions) < 1:
            # we extracted all acronyms from text or links
            return collections.OrderedDict(sorted(expansion_dict.items())), links_followed if links_followed else []

        # replace known expansions in text, this might help with context
        for acronym, expansion in expansion_dict.items():
            if expansion is not None:
                expansion = expansion[0]
                raw_text = re.sub(
                    "\\b" + re.escape(acronym) + "\\b",
                    re.escape(expansion),
                    raw_text,
                    re.IGNORECASE,
                )

        article.set_raw_text(raw_text=raw_text)
        article.set_preprocessor(
            lambda text: self.text_preprocessor(text, acronyms_without_expansions)
        )
        # preprocessed_text = self.text_preprocessor(text)

        out_expander_output = self.out_expander.process_article(
            article, acronyms_without_expansions, test_article_id
        )

        out_expansions = {
            k: ([v, ExpansionModuleEnum.out_expander] if v is not None else v)
            for k, v in out_expander_output.items()
        }

        expansion_dict.update(out_expansions)

        # Removes acronyms with no expansion found
        expansion_dict = {
            acronym: expansion
            for acronym, expansion in expansion_dict.items()
            if expansion is not None
        }
        return collections.OrderedDict(sorted(expansion_dict.items())), links_followed if links_followed else []


class AcronymExpanderFactory:  # pylint: disable=too-few-public-methods
    """
    Factory to create expander for acronyms found in text either in or out
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        text_preprocessor: Callable = text_preprocessing,
        in_expander_name: str = "schwartz_hearst_original",
        in_expander_args=None,
        out_expander_name: str = "random",
        out_expander_args=None,
        follow_links: bool = False,
        follow_links_cache: bool = False,
        bypass_db: bool = False,
        run_config: RunConfig = RunConfig(),
    ):
        """
        Args:
            text_preprocessor: text preprocessor function
            in_expander_name: In expander technique name
            in_expander_args: Arguments for the in expander
            out_expander_name: Out expander technique name
            out_expander_args: Arguments for the out expander
            follow_links: follow links factory to create a follow links object
                that process html tags and extract acronyms from those links
            run_config: general run configurations
        """

        self.text_preprocessor = text_preprocessor

        self.in_expander_name = in_expander_name
        self.in_expander_args = in_expander_args

        self.acronym_out_expander_factory = AcronymOutExpanderFactory(
            out_expander_name=out_expander_name,
            out_expander_args=out_expander_args,
            run_config=run_config,
        )

        self.follow_links = follow_links
        self.follow_links_cache = follow_links_cache

        self.bypass_db = bypass_db

        self.run_config = run_config

    def _get_in_expander(self, train_data_manager_in_expander: TrainInDataManager):

        if self.in_expander_args is None:
            self.in_expander_args = {}

        self.in_expander_args[
            "train_data_manager_base"
        ] = train_data_manager_in_expander

        if self.in_expander_name == "schwartz_hearst_abbreviations":
            return AcroExpExtractor_Schwartz_Hearst()
        elif self.in_expander_name == "schwartz_hearst_original":
            return AcroExpExtractor_Original_Schwartz_Hearst()
        elif self.in_expander_name == "ours":
            return AcroExpExtractor_Yet_Another_Improvement()
        elif self.in_expander_name == "schwartz_hearst":
            return AcroExpExtractor_AAAI_Schwartz_Hearst()
        elif self.in_expander_name == "maddog":
            from AcroExpExtractors.AcroExpExtractor_MadDog import (
                AcroExpExtractor_MadDog,
            )
            return AcroExpExtractor_MadDog()
        elif self.in_expander_name == "scibert_sklearn":
            from AcroExpExtractors.AcroExpExtractor_Scibert_Sklearn import (
                AcroExpExtractor_Scibert_Sklearn_Factory,
            )
            factory = AcroExpExtractor_Scibert_Sklearn_Factory(**self.in_expander_args)
            return factory.get_in_expander()
        elif self.in_expander_name == "scibert_allennlp":
            from AcroExpExtractors.AcroExpExtractor_Scibert_Allennlp import (
                AcroExpExtractor_Scibert_Allennlp_Factory,
            )
            factory = AcroExpExtractor_Scibert_Allennlp_Factory(**self.in_expander_args)
            return factory.get_in_expander()
            
        elif self.in_expander_name == "sci_dr":
            from AcroExpExtractors.AcroExpExtractor_Sci_Dr import AcroExpExtractor_Sci_Dr_Factory
            factory = AcroExpExtractor_Sci_Dr_Factory(**self.in_expander_args)
            return factory.get_in_expander()
        else:
            raise ValueError(
                "No extractor found with name: {}".format(self.in_expander_name)
            )

    def create_expander(
        self,
        train_data_manager_out_expander: TrainOutDataManager,
        train_data_manager_in_expander: TrainInDataManager,
    ) -> AcronymExpander:
        """
        Create an AcronymExpander object based on the data provided by the train_data_manager_out_expander
         that will assign expansions to acronyms found in text

        Args:
            train_data_manager_out_expander: manager of the data to create the out expander
        Returns:
            Acronym Expander trained on the data provided by the train_data_manager
        """
        text_preprocessor = self.text_preprocessor

        in_expander = self._get_in_expander(train_data_manager_in_expander)

        (
            acronym_out_expander,
            out_expander_model_execution_time,
        ) = self.acronym_out_expander_factory.create_out_expander(
            train_data_manager_out_expander, bypass_db=self.bypass_db
        )

        if self.follow_links:
            links_follower = LinksFollower(in_expander, self.follow_links_cache).process
        else:
            links_follower = None

        acronym_expander = AcronymExpander(
            text_preprocessor=text_preprocessor,
            in_expander=in_expander,
            acronym_out_expander=acronym_out_expander,
            links_follower=links_follower,
        )

        return acronym_expander, out_expander_model_execution_time
