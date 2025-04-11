# first line: 1
@memory.cache
def load_session_cached(patient_list):
    return check_session(loader.load_session_data(patient_list=patient_list))
