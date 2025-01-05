from enum import Enum

class LossName(Enum):
    ECE = 'ECE'
    adaECE = 'adaECE'
    NLL = 'NLL'
    adaNLL = 'adaNLL'
    BS = 'BS'
    adaBS = 'adaBS'
    SCE = 'SCE'
    adaSCE = 'adaSCE'