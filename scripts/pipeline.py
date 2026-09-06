from download_data import download_dataset 
from data_exploration import explore_data
from filter_data import filter_ticker_streaming
def main():
    download_dataset()
    print('\n\n first step done! dataset downloaded to config raw_data_path')
    explore_data()
    print('\n\n second step done! info abt dataset extracted')
    filter_ticker_streaming()
    print('\n\n third step done! filtered data saved to config processed_data_path')

if __name__=='__main__':
    main()