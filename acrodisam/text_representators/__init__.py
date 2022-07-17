"""
Text Representators
"""
from abc import ABCMeta, abstractmethod
import importlib
import inspect
import os
from typing import List

from ._base import TextRepresentatorFactory, TextRepresentator


IMP_MODULE = "text_representators.impl"

ERROR_MESSAGE_IMPORTING = (
    "Text representator file {}.py was found but we got the"
    " following error when importing: \n {}: {}."
)


def get_available_text_representators() -> List[str]:
    """
    Returns the list of the available text representator names
    """

    list_modules = []
    this_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    dir_path = os.path.dirname(this_path)  # script directory
    dir_path += os.sep + "impl"
    for module in os.listdir(dir_path):
        if module == "__init__.py" or module[-3:] != ".py":
            continue
        list_modules.append(module[:-3])
    return list_modules


def get_text_representator_factory(
    text_representator: str, *args, **kw
) -> TextRepresentatorFactory:
    """
    Imports and instantiates a Text Representator Factory

    :param text_representator: text representator name
    :param *args lists of arguments to pass to the text representator factory
    :param **kw dict of keyword arguments to pass to the text representator factory
    :raises ValueError: if unsupported or incorrect values are passed as arguments
    :raises TypeError: if incorrect argument types
    :raises Exception: remaining errors, e.g., import issues
    """
    # import the text representator module
    module_name = IMP_MODULE + "." + text_representator
    try:
        mod = importlib.import_module(module_name, package=text_representator)
    except ModuleNotFoundError as exception:
        if exception.name == module_name:
            representators = get_available_text_representators()
            raise ModuleNotFoundError(
                "Text representator {} not found! "
                "Available text representators are: {{{}}}".format(
                    text_representator, ",".join(representators)
                )
            ) from exception

        raise ValueError(
            ERROR_MESSAGE_IMPORTING.format(
                text_representator, type(exception).__name__, exception
            )
        ) from exception

    except Exception as exception:
        raise ValueError(
            ERROR_MESSAGE_IMPORTING.format(
                text_representator, type(exception).__name__, exception
            )
        ) from exception

    # First we test if obj is defined in the imported module, otherwise we skip.
    predicate = (
        lambda obj: inspect.getmodule(obj) == mod
        and inspect.isclass(obj)
        and issubclass(obj, TextRepresentatorFactory)
    )

    for _, obj in inspect.getmembers(mod, predicate):
        try:
            return obj(*args, **kw)
        except (ValueError, TypeError) as ex:  # as e:
            message = (
                "Unable to create {} text representator, invalid parameter(s)"
                " in args={} and kwargs={} \n due to: \n {}".format(
                    text_representator, args, kw, ex
                )
            )

            if isinstance(ex, ValueError):
                raise ValueError(message) from ex
            raise TypeError(message) from ex

        except Exception as ex:
            raise Exception(
                "Failed to create text representator with name={}"
                " and parameters args={} and kwargs={} \n due to:"
                " \n {}".format(text_representator, args, kw, ex)
            ) from ex

    raise ValueError(
        "Unable to create text representator {0}, valid implementation of"
        " OutExpanderFactory class was not found in {0}.py file.".format(
            text_representator
        )
    )
