from data_exploration import explore_data
from filter_data import filter_ticker_streaming


def main():
    # explore_data()
    # print('\n\nstep done! info abt dataset extracted')
    filter_ticker_streaming()
    print("\n\nstep done! filtered data saved to config processed_data_path")
    explore_data(data_path="data/processed/nvda_news.csv")


if __name__ == "__main__":
    main()
