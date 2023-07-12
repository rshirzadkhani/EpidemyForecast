from enum import Enum, auto


class Data(Enum):
    WIFI = auto()
    SCHOOL = auto()
    SAFEGRAPH = auto()
    WORKPLACE = auto()
    LYONSCHOOL = auto()
    HIGHSCHOOL = auto()
    CONFERENCE = auto()

class Network(Enum):
    BASE = auto()
    ER = auto()
    ER_TEMPORAL = auto()
    BA = auto()
    Regular = auto()
    TEMPORAL = auto()
    STATIC = auto()
    MST_W_MATCH = auto()
    MST_D_MATCH = auto()
    MST_D_MATCH_2 = auto()
    STANDARD_EQ = auto()
    STANDARD_GRAPH = auto()