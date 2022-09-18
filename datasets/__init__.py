from .sunrefer import SUNREFER3D

def build_dataset(args, isTrain):
    return SUNREFER3D(args, isTrain)
    