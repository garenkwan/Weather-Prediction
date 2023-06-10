import logging, yaml#, time
import pandas as pd
import numpy as np

from scripts.data_download import append_zero

with open("scripts/station_ids.yml", "r") as stream:
    station_ids = yaml.safe_load(stream)

class weatherData:
    def __init__(self, 
                 start_date="2021-10-15",
                 end_date="2021-10-18", 
                 training_window=25, 
                 forecast=24):
        self.start_date=start_date
        self.end_date=end_date
        self.training_window=training_window
        self.forecast=forecast
        self.station_ids = station_ids

    def data_fill_missing(self, df, date):
        hours = set([x for x in range(24)])
        cols = set([x.hour for x in df.columns])
        missing = hours.difference(cols)
        date = pd.to_datetime(date)
        if len(missing) > 0 and len(df.columns) > 0:
            logging.debug('activate data filling')
            for x in sorted(list(missing)):
                if x != 0:
                    logging.debug('forward filling')
                    add_date = date + pd.to_timedelta(x, unit='h')
                    previous_date = date + pd.to_timedelta(x-1, unit='h')
                    df[add_date] = df[previous_date]
                else:
                    logging.debug('missing 0')
                    df[date] = df[df.columns[0]]
        elif len(df.columns) == 0:
            logging.debug('all missing')
            columns = [date + pd.to_timedelta(x, unit='h') for x in range(24)]
            df[columns] = 0
        ordered_columns = [date + pd.to_timedelta(x, unit='h') for x in range(24)]
        return df[ordered_columns]
        
    def data_transpose(self, df, data_type):
        df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time'], unit='h')
        df_T = df.loc[(df['data_type']==data_type), ['station_id', 'timestamp', 'value']]\
                .pivot(index='station_id', columns='timestamp', values='value')
        return df_T
        
    def df_preprocessing(self, df, data_type):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].apply(lambda x: x.date)
        df['time'] = df['timestamp'].apply(lambda x: x.hour)
        groups = ['station_id', 'name', 'data_type', 'date', 'time', 'value']

        if data_type == 'environment/rainfall':
            df_grouped = df[groups].groupby(groups[:-1]).sum().reset_index()\
            .sort_values(['date', 'time', 'station_id', 'data_type'])
        else:
            df_grouped = df[groups].groupby(groups[:-1]).mean().reset_index()\
            .sort_values(['date', 'time', 'station_id', 'data_type'])
        return df_grouped

    def get_data(self, date):
        data_types = ['environment/rainfall', 
                    'environment/relative-humidity',
                    'environment/wind-direction', 
                    'environment/wind-speed',
                    'environment/air-temperature']
        data = pd.read_csv("./data/"+date+".csv")

        result = []
        for data_type in data_types:
            df_index = pd.DataFrame(self.station_ids, columns=['station_id']).set_index('station_id')
            df_process = self.df_preprocessing(data, data_type)
            df_T = self.data_transpose(df_process, data_type)            
            df_T = self.data_fill_missing(df_T, date)
            final = df_index.join(df_T).fillna(0.0)
            result.append(final)
        return result

    def training_data(self, start_date, end_date, training_window, forecast):
        '''
        Prepares a list of 3D arrays and predictor
        training_window: number of hours that should be used X
        forecast: number of hourst that should be used as Y
        '''

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        window_size = training_window + forecast

        training_data = []
        predict_data = []
        
        days = (end_date - start_date).days
        date_range = [(start_date + pd.to_timedelta(x, unit='d')).strftime('%Y-%m-%d')\
                    for x in range(days+1)]
        
        days_per_window = window_size // 24

        # This should prepare the basic df
        if days_per_window > 0 and days_per_window < days:
            df_list = self.get_data(date_range.pop(0))
            days_per_window -= 1
            days -= 1
            while days_per_window >= 0:
                extend_list = self.get_data(date_range.pop(0))
                for i, df in enumerate(extend_list):
                    df_list[i] = pd.concat([df_list[0], df], axis=1)
                days_per_window -= 1
                days -= 1
        else:
            df_list = self.get_data(date_range.pop(0))
        
        logging.debug(date_range)
        logging.debug('Number of layers:\t{}\nSize of df:\t'.format(len(df_list), df_list[0].shape))

        # Perhaps training can start here
        working_columns = list(df_list[0].columns)
        while days > 0 or len(working_columns) >= window_size:
            while window_size <= len(working_columns):
                train_columns = working_columns[:training_window]
                predict_column = working_columns[training_window:training_window+forecast]
                
                training_data = np.array(
                                                [
                                                np.array(df_list[0][train_columns]),
                                                np.array(df_list[1][train_columns]),
                                                np.array(df_list[2][train_columns]),
                                                np.array(df_list[3][train_columns]),
                                                np.array(df_list[4][train_columns])
                                                ]
                                            )
                predict_data = np.sum(np.array(df_list[0][predict_column]), axis=1)
            
                working_columns.pop(0)
                yield training_data, predict_data

            if days > 0:
                extend_list = self.get_data(date_range.pop(0))
                for i, df in enumerate(extend_list):
                    df_list[i] = pd.concat([df_list[0], df], axis=1)
                working_columns = list(df_list[0].columns)
                days -= 1

# if __name__ == '__main__':
#     time_format = time.ctime().split()
#     timestamp = [time_format[-1], time_format[1], append_zero(int(time_format[2])), time_format[3].replace(':','')]
#     filename = './logs/processing_' + '_'.join(timestamp) + '.log'
#     logging.basicConfig(filename=filename,
#                         format='%(asctime)s %(message)s', 
#                         datefmt='%m/%d/%Y %I:%M:%S %p',
#                         encoding='utf-8', 
#                         level=logging.INFO)
#     dataset = weatherData()
#     i = 0
#     for training, predict in dataset.training_data("2021-10-15", "2021-10-18", 25, 24):
#         logging.info('\nTRAINING\nFrame index:\t{}\nSize of array:\t{}'.format(i, training.shape))
#         logging.info('\nPREDICT\nFrame index:\t{}\nSize of array:\t{}'.format(i, predict.shape))