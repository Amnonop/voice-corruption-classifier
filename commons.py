CONFIGS_DIR = 'configs/'
STATES_DIR = 'states/'

class_ids = {'M': 0, 'F': 1}


def get_class_name(class_id):
    for key, value in class_ids.items():
        if value == class_id:
            return key
