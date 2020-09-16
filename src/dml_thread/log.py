import logging
# import warnings

logger = logging.getLogger('DML E-Chem')
logging.basicConfig(
    level=logging.INFO,  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%d/%m/%Y %I:%M:%S %p',
    filename='./logs/dmlmung.log',

    filemode='w'  # "a"
)

# logging.debug('This message should go to the log file')
# logging.warning('Watch out!')  # will print a message to the console
# logging.info('I told you so')  # will not print anything
