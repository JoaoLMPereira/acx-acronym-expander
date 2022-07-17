import enum


class ExpansionModuleEnum(enum.Enum):
    """Represents the expansion module that is used by the Acronym Expander system to expand acronyms"""

    in_expander = 1
    link_follower = 2
    out_expander = 3