from enum import Enum


class RelationType(Enum):
    Cast = "Cast"
    Call = "Call"
    Import = "Import"
    Set = "Set"
    CallNonDynamic = "Call non-dynamic"
    Implement = "Implement"
    Modify = "Modify"
    Annotate = "Annotate"
    UseVar = "UseVar"
    Typed = "Typed"
    Define = "Define"
    Reflect = "Reflect"
    Parameter = "Parameter"
    Override = "Override"
    Inherit = "Inherit"
    Contain = "Contain"
