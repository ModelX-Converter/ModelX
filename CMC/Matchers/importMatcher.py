from CMC.utils import log_info, search_pytorch_to_paddle_api
from CMC.special_mapping import SPECIALMODULE_MAPPING


def pytorch_to_paddlepaddle_replace_Import(aliasName, sourceDLClass, targetDLClass, logger):
    # special module in pytorch to paddlepaddle

    for module in SPECIALMODULE_MAPPING[sourceDLClass]:
        if module in aliasName and targetDLClass in SPECIALMODULE_MAPPING[sourceDLClass][module]:
            aliasName = aliasName.replace(module, SPECIALMODULE_MAPPING[sourceDLClass][module][targetDLClass])
            log_info(logger, "according to Special mapping, converting paddle module: {}.".format(
                aliasName))
    return search_pytorch_to_paddle_api(aliasName, logger)
