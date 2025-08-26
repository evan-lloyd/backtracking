from enum import StrEnum


class DatasetStage(StrEnum):
    raw = "raw"
    token = "token"
    match = "match"
    activation = "activation"
    classification = "classification"
