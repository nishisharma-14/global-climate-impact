import logging
import logging.config

LOG_CONFIG = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard'
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO'
    }
}


def configure_logging():
    logging.config.dictConfig(LOG_CONFIG)


if __name__ == '__main__':
    configure_logging()
    logging.getLogger(__name__).info('Logging configured')
