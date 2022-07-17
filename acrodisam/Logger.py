import os
import logging
import logging.config

import yaml

from string_constants import FOLDER_ROOT, FILE_LOG_CONF, file_logs, name_logger



default_path=FILE_LOG_CONF
default_level=logging.INFO
env_key='LOG_CFG'


def fixFilePathsAux(key, value):
    # path = os.path.dirname(os.path.realpath(__file__))
    #spath = os.path.join(path, 'bot.log')
    if isinstance(value, dict):
        return {k:fixFilePathsAux(k, v) for (k, v) in value.items()}
    
    if key.lower() == "filename":
        return FOLDER_ROOT + value
    
    return value
        

def fixFilePaths(configDict):
    return fixFilePathsAux("root", configDict)


path = default_path
value = os.getenv(env_key, None)
if value:
    path = value
if os.path.exists(path):
    with open(path, 'rt') as f:
        config = yaml.safe_load(f.read())
        fixedConfig = fixFilePaths(config)
    logging.config.dictConfig(fixedConfig)
else:
    logging_format = "%(levelname)s: %(process)d %(asctime)s %(message)s"
    logging.basicConfig(filename=file_logs,
        level=default_level, format=logging_format, datefmt='%m/%d/%Y %I:%M:%S %p')
        
# This enables logs inside process

common_logger = logging.getLogger(name_logger)
common_logger.critical("started logger")

