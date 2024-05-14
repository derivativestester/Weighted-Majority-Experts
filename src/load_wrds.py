import pandas as pd
from pandas.tseries.offsets import MonthEnd, YearEnd
import numpy as np
import wrds
import config
from pathlib import Path


OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)
WRDS_USERNAME = config.WRDS_USERNAME

def load_13f(wrds_username=WRDS_USERNAME):
    sql_query = """
        SELECT  *
        FROM wrdssec.wrds_13f_holdings
        WHERE
            fdate >= '2012-01-01'
            AND fdate <= '2023-12-29'
            AND (cik = '0001649339'
            OR cik = '0001067983'
            OR cik = '0001536411'
            OR cik = '0001657335'
            OR cik = '0001336528'
            OR cik = '0001037389'
            OR cik = '0001791786'
            OR cik = '0001541617'
            OR cik = '0001079114'
            OR cik = '0001553733'
            OR cik = '0001056823'
            OR cik = '0000814375'
            OR cik = '0001050464'
            OR cik = '0001169883'
            OR cik = '0001541901')
        """
    db = wrds.Connection(wrds_username=wrds_username)
    mngr_data = db.raw_sql(sql_query, date_cols=["fdate"])
    db.close()

    return mngr_data

def load_daily_stock_returns(wrds_username=WRDS_USERNAME):
    sql_query = """
        SELECT  cusip, prc, ret, date
        FROM crsp_a_stock.dsf
        WHERE date >= '2012-01-01'
        AND date <= '2023-12-29'
        """
    db = wrds.Connection(wrds_username=wrds_username)
    stock_returns = db.raw_sql(sql_query, date_cols=["date"])
    db.close()

    return stock_returns

def load_combined_grouped(df, df2):
    # reset index to the fdate quarter and add 45 days to the quarter
    df['new_fdate'] = df['fdate'].dt.to_period('Q').dt.to_timestamp() + pd.DateOffset(days=45)
    # next fdate (next quarter) and add 45 days to the quarter
    df['next_fdate'] = df['new_fdate'] + pd.DateOffset(months=3)

    # drop any columns that do not have 41 unique new_fdate
    df = df[df.groupby('cik')['new_fdate'].transform('nunique') == 41]

    # drop last digit of cusip
    df['cusip'] = df['cusip'].str[:-1]

    #find the cusips that are in both dataframes
    cusips = np.intersect1d(df['cusip'].unique(), df2['cusip'].unique())

    #drop rows with cusips that are not in both dataframes
    df = df[df['cusip'].isin(cusips)]
    df2 = df2[df2['cusip'].isin(cusips)]

    # create a new column for compounded return for each daily date
    df2 = df2.sort_values(by=['cusip', 'date'])
    df2['compounded'] = (1 + df2['ret']).groupby(df2['cusip']).cumprod()

    # Calculate 3-month rolling standard deviation for each 'cusip'
    df2['rolling_std_3m'] = df2.groupby('cusip')['compounded'].rolling(63).std().reset_index(drop=True)

    # Find missing dates in df2 for each cusip in df
    missing_new_fdates = df[~df['new_fdate'].isin(df2['date'])][['cusip', 'new_fdate']]
    missing_next_fdates = df[~df['next_fdate'].isin(df2['date'])][['cusip', 'next_fdate']]
    # rename columns
    missing_new_fdates.columns = ['cusip', 'date']
    missing_next_fdates.columns = ['cusip', 'date']
    missing = pd.concat([missing_new_fdates, missing_next_fdates])
    missing.drop_duplicates(subset=['cusip', 'date'], inplace=True)
    missing

    # append missing into df2 with rest of the columsn as NaN
    missing['ret'] = np.nan
    missing['compounded'] = np.nan
    missing['rolling_std_3m'] = np.nan
    df2 = pd.concat([df2, missing], axis=0)
    df2 = df2.sort_values(by=['cusip', 'date'])
    df2.sort_values(by=['cusip', 'date'], inplace=True)
    df2.reset_index(drop=True, inplace=True)

    #forward fill the missing values for prc, compounded, and rolling_std_3m
    df2['prc'] = df2.groupby('cusip')['prc'].ffill()
    df2['compounded'] = df2.groupby('cusip')['compounded'].ffill()
    df2['rolling_std_3m'] = df2.groupby('cusip')['rolling_std_3m'].ffill()

    # add prc to the df, with date same as new_fdate and cusip same as cusip
    df = pd.merge(df, df2[['date', 'cusip', 'prc', 'compounded']], left_on=['new_fdate', 'cusip'], right_on=['date', 'cusip'], how='left')
    # add prc to the df, with date same as next_fdate and cusip same as cusip
    df = pd.merge(df, df2[['date', 'cusip', 'prc', 'compounded', 'rolling_std_3m']], left_on=['next_fdate', 'cusip'], right_on=['date', 'cusip'], how='left')
    # drop NaN prc values
    df = df.dropna(subset=['prc_x', 'prc_y'])
    # find performance by taking compounded return of next_fdate divided by compounded return of new_fdate
    df['performance'] = df['compounded_y'] / df['compounded_x'] - 1

    # for each cik, calculate the weighted performance using value as weight.
    # also calculate weighted risk to reward ratio by taking performance / rolling_std_3m
    df['weight'] = df.groupby(['cik', 'new_fdate'])['value'].transform(lambda x: x / x.sum())
    df['weighted_performance'] = df['performance'] * df['weight']
    df['weighted_risk'] = df['rolling_std_3m'] * df['weight']

    # group by cik and sum the weighted performance and weighted risk to reward ratio for each new_fdate
    df_grouped = df.groupby(['cik', 'new_fdate'])[['weighted_performance', 'weighted_risk']].sum().reset_index()
    df_grouped['risk_reward_ratio'] = df_grouped['weighted_performance'] / df_grouped['weighted_risk']
    return df_grouped