"""
Term Representators
"""
from abc import ABCMeta, abstractmethod
import importlib
import inspect
import os
from typing import List

from ._base import TermRepresentatorFactory, TermRepresentator


IMP_MODULE = "term_representators.impl"

ERROR_MESSAGE_IMPORTING = (
    "Term representator file {}.py was found but we got the"
    " following error when importing: \n {}: {}."
)

CUSTOM_TERM_REPRESENTATORS = {
    "classic_context_vector": ["text_to_term", "document_context_vector"],
    "sbe": ["text_to_term", "sbe_text"],
}


def get_available_term_representators() -> List[str]:
    """
    Returns the list of the available term representator names
    """

    list_modules = []
    this_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    dir_path = os.path.dirname(this_path)  # script directory
    dir_path += os.sep + "impl"
    for module in os.listdir(dir_path):
        if module == "__init__.py" or module[-3:] != ".py":
            continue
        list_modules.append(module[:-3])
    return list_modules + list(CUSTOM_TERM_REPRESENTATORS.keys())


def get_term_representator_factory(
    term_representator: str, *args, **kw
) -> TermRepresentatorFactory:
    """
    Imports and instantiates a Term Representator Factory

    Args:
        term_representator: term representator name
        *args lists of arguments to pass to the term representator factory
        **kw dict of keyword arguments to pass to the term representator factory

    Raises:
        ValueError: if unsupported or incorrect values are passed as arguments
        TypeError: if incorrect argument types
        Exception: remaining errors, e.g., import issues
    """
    custom_rep = CUSTOM_TERM_REPRESENTATORS.get(term_representator)
    if custom_rep:
        term_representator = custom_rep[0]
        args = custom_rep[1:] + list(args)

    # import the text representator module
    module_name = IMP_MODULE + "." + term_representator
    try:
        mod = importlib.import_module(module_name, package=term_representator)
    except ModuleNotFoundError as exception:
        if exception.name == module_name:
            representators = get_available_term_representators()
            raise ModuleNotFoundError(
                "Term representator {} not found! "
                "Available term representators are: {{{}}}".format(
                    term_representator, ",".join(representators)
                )
            ) from exception

        raise ValueError(
            ERROR_MESSAGE_IMPORTING.format(
                term_representator, type(exception).__name__, exception
            )
        ) from exception

    except Exception as exception:
        raise ValueError(
            ERROR_MESSAGE_IMPORTING.format(
                term_representator, type(exception).__name__, exception
            )
        ) from exception

    # First we test if obj is defined in the imported module, otherwise we skip.
    predicate = (
        lambda obj: inspect.getmodule(obj) == mod
        and inspect.isclass(obj)
        and issubclass(obj, TermRepresentatorFactory)
    )

    for _, obj in inspect.getmembers(mod, predicate):
        try:
            return obj(*args, **kw)
        except (ValueError, TypeError) as ex:  # as e:
            message = (
                "Unable to create {} term representator, invalid parameter(s)"
                " in args={} and kwargs={} \n due to: \n {}".format(
                    term_representator, args, kw, ex
                )
            )

            if isinstance(ex, ValueError):
                raise ValueError(message) from ex
            raise TypeError(message) from ex

        except Exception as ex:
            raise Exception(
                "Failed to create term representator with name={}"
                " and parameters args={} and kwargs={} \n due to:"
                " \n {}".format(term_representator, args, kw, ex)
            ) from ex

    raise ValueError(
        "Unable to create term representator {0}, valid implementation of"
        " OutExpanderFactory class was not found in {0}.py file.".format(
            term_representator
        )
    )
