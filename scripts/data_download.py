import json, logging, os, requests, time, threading, tqdm
import pandas as pd

def append_zero(num:int):
    if num < 10:
        return '0' + str(num)
    else:
        return str(num)

def attempt_get(get_url, params):
    max_tries = counter = 3
    
    logging.info('Attempt {} of {} for {} on {}'.\
                 format(max_tries-counter+1,
                        counter,
                        get_url,
                        params['date']))
    result = requests.get(get_url, params=params)
    while result.status_code != 200 and counter > 0:
        counter -= 1
        time.sleep(1)
        result = requests.get(get_url, params=params)
    if result.status_code != 200:
        return 'Failed'
    else:
        return result.json()
    
def collect_data(date_range:list):
    base_url = "https://api.data.gov.sg/v1"
    wind_speed_url = "/environment/wind-speed"
    wind_direction_url = "/environment/wind-direction"
    rh_url = "/environment/relative-humidity"
    rainfall_url = "/environment/rainfall"
    air_temp_url = "/environment/air-temperature"
    
    urls = [wind_speed_url, wind_direction_url,
            rh_url, rainfall_url, air_temp_url]
    
    failures = []
    
    for date in date_range:
        results_df = pd.DataFrame([])
        for url in urls:
            json_result = get_data(base_url + url, date)
            partial_df = process_data(json_result)
            if type(partial_df) != str:
                partial_df['data_type'] = url
                results_df = pd.concat([results_df, partial_df])
            else:
                failures.append([date, url])
        results_df = results_df.reset_index(drop=True)
        results_path = './data/'+date+'.csv'
        results_df.to_csv(path_or_buf=results_path,
                          index=False,
                          mode='w+')
        logging.info('Wrote to {}'.format(results_path))
    time.sleep(1)
    if len(failures) > 0:
        date_range = [x[0] for x in failures]
        collect_data(date_range)
        
def get_and_process_data(url, date, list_of_dfs):
    base_url = "https://api.data.gov.sg/v1"
    json_file = get_data(url, date)
    if type(json_file) != str:
        df = process_data(json_file)
        data_type = url.lstrip(base_url)
        df['data_type'] = data_type
    list_of_dfs.append(df)
    
def collect_data_threaded(date_range:list):
    base_url = "https://api.data.gov.sg/v1"
    wind_speed_url = "/environment/wind-speed"
    wind_direction_url = "/environment/wind-direction"
    rh_url = "/environment/relative-humidity"
    rainfall_url = "/environment/rainfall"
    air_temp_url = "/environment/air-temperature"
    
    urls = [wind_speed_url, wind_direction_url,
            rh_url, rainfall_url, air_temp_url]
    
    failures = []
    
    for date in tqdm.tqdm(date_range):
        threads = []
        list_of_dfs = []
        results_df = pd.DataFrame([])
        for url in urls:
            process = threading.Thread(target=get_and_process_data, 
                                       args=(base_url+url, date, list_of_dfs))
            process.start()
            threads.append(process)
        for process in threads:
            process.join()
            
        completed = True
        for df in list_of_dfs:
            if type(df) == str:
                completed = False
        if completed:
            for df in list_of_dfs:
                results_df = pd.concat([results_df, df])
        else:
            failures.append(date)
        results_df = results_df.reset_index(drop=True)
        results_path = './data/'+date+'.csv'
        results_df.to_csv(path_or_buf=results_path,
                          index=False,
                          mode='w+')
        logging.info('Wrote to {}'.format(results_path))
        time.sleep(1)
    if len(failures) > 0:
        collect_data_threaded(failures)

def get_data(url:str, date:str):
    '''
    Takes url and date, both strings
    Returns result, a json file
    '''
    params = {'date': date}
    result = attempt_get(url, params=params)
    if result == 'Failed':
        logging.info('Data collection for {} on {} failed.'.format(url,
                                                                   params['date']))
        return params['date']
    else:
        return result
    
def get_dates(start_date='2017-01-01', end_date='2023-12-31'):
    '''
    Function will download data for whole month within date range.
    Further functionality is neither required nor worth the effort.
    '''
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    dates = []
    for x in range(start_date.year, end_date.year+1):
        for y in range(start_date.month, end_date.month+1):
            for z in range(1, 32):
                if y == 2 and z > 29 and x == 2020:
                    continue
                elif y == 2 and z > 28 and x != 2020:
                    continue
                elif y in [4, 6, 9, 11] and z == 31:
                    continue
                else:
                    dates.append(str(x) + '-' + append_zero(y) + '-' + append_zero(z))
    return dates
        
def process_data(json_file):
    if len(json_file['metadata']['stations']) > 0:
        metadata = pd.DataFrame.from_records(json_file['metadata']['stations'])
        metadata['latitude'] = metadata['location'].apply(lambda x: x['latitude'])
        metadata['longitude'] = metadata['location'].apply(lambda x: x['longitude'])
        metadata = metadata.drop(['id','location'], axis=1)
        metadata = metadata.set_index('device_id')

        results_df = pd.DataFrame([])
        for data_row in json_file['items']:
            timestamp = pd.to_datetime(data_row['timestamp'])
            readings = data_row['readings']
            df = pd.DataFrame(readings)
            df['timestamp'] = timestamp
            results_df = pd.concat([results_df, df])
        results_df = results_df.set_index('station_id')
        results_df = results_df.join(metadata).reset_index()
        column_names = [x if x != 'index' else 'station_id' for x in list(results_df.columns)]
        results_df.columns = column_names
        return results_df
    else:
        return pd.DataFrame([])
    
    
if __name__ == '__main__':
    time_format = time.ctime().split()
    timestamp = [time_format[-1], time_format[1], append_zero(int(time_format[2])), time_format[3].replace(':','')]
    filename = './logs/download_' + '_'.join(timestamp) + '.log'
    logging.basicConfig(filename=filename,
                        format='%(asctime)s %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        encoding='utf-8', 
                        level=logging.INFO)
    dates = get_dates()
    collect_data_threaded(dates[-1:])