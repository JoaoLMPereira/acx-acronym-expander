import importlib
import inspect
import os

from out_expanders._base import OutExpander, OutExpanderFactory, OutExpanderArticleInput


IMP_MODULE = "out_expanders.impl"

ERROR_MESSAGE_IMPORTING = (
    "Out-expander file {}.py was found but we got the"
    " following error when importing: \n {}: {}."
)


def get_available_out_expanders():
    list_modules = []

    dir_path = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )  # script directory
    dir_path += os.sep + "impl"
    for module in os.listdir(dir_path):
        if module == "__init__.py" or module[-3:] != ".py":
            continue
        list_modules.append(module[:-3])
    return list_modules


def get_out_expander_factory(out_expander: str, *args, **kw) -> OutExpanderFactory:
    # import the out-expander module
    module_name = IMP_MODULE + "." + out_expander
    try:
        mod = importlib.import_module(module_name, package=out_expander)
    except ModuleNotFoundError as ex:
        if ex.name == module_name:
            available_out_expanders = get_available_out_expanders()
            raise ValueError(
                "Out-expander {} not found! Available out-expanders are: {{{}}}".format(
                    out_expander, ",".join(available_out_expanders)
                )
            ) from ex

        raise ValueError(
            ERROR_MESSAGE_IMPORTING.format(out_expander, type(ex).__name__, ex)
        ) from ex
    except Exception as ex:
        raise ValueError(
            ERROR_MESSAGE_IMPORTING.format(out_expander, type(ex).__name__, ex)
        ) from ex

    # First we test if obj is defined in the imported module, otherwise we skip.
    predicate = (
        lambda obj: inspect.getmodule(obj) == mod
        and inspect.isclass(obj)
        and issubclass(obj, OutExpanderFactory)
    )
    for _, obj in inspect.getmembers(mod, predicate):
        try:
            return obj(*args, **kw)
        except (ValueError, TypeError) as ex:  # as e:
            message = (
                "Unable to create {} out-expander, invalid parameter(s)"
                " in args={} and kwargs={} \n due to: \n {}".format(
                    out_expander, args, kw, ex
                )
            )

            if isinstance(ex, ValueError):
                raise ValueError(message) from ex
            raise TypeError(message) from ex
        except Exception as ex:
            raise Exception(
                "Failed to create out-expander with name={}"
                " and parameters args={} and kwargs={} \n due to:"
                " \n {}".format(out_expander, args, kw, ex)
            ) from ex

    raise ValueError(
        "Unable to create out-expander {0}, valid implementation of"
        " OutExpanderFactory class was not found in {0}.py file.".format(out_expander)
    )
