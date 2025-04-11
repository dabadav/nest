# first line: 1
@memory.cache
def load_timeseries_cached(patient_list):
    return loader.load_timeseries_data(patient_list=patient_list)
