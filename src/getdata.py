import pandas as pd
import math


def readcsv(address):
    df = pd.read_csv(address, encoding='gb18030')
    df['Rating'] = df['Rating'].astype(str)
    df['Score_taste'] = df['Score_taste'].astype(str)
    df['Score_environment'] = df['Score_environment'].astype(str)
    df['Score_service'] = df['Score_service'].astype(str)
    # 去除'Rating', 'Score_taste', 'Score_environment', 'Score_service'值为"|"符号和"产"“场”“房”字样的行
    df = df[~df['Rating'].str.contains('\|')]
    df = df[~df['Score_taste'].str.contains('\|')]
    df = df[~df['Score_environment'].str.contains('\|')]
    df = df[~df['Score_service'].str.contains('\|')]
    df = df[~df['Rating'].str.contains('产|场|房')]
    df = df[~df['Score_taste'].str.contains('产|场|房')]
    df = df[~df['Score_environment'].str.contains('产|场|房')]
    df = df[~df['Score_service'].str.contains('产|场|房')]
    # 'Rating', 'Score_taste', 'Score_environment', 'Score_service'转换为int类型
    df['Rating'] = df['Rating'].astype(int)
    df['Score_taste'] = df['Score_taste'].astype(int)
    df['Score_environment'] = df['Score_environment'].astype(int)
    df['Score_service'] = df['Score_service'].astype(int)
    # 将Time中不含有年份设置为2017年
    df['Time'] = df['Time'].str.replace('月', '/').str.replace('日', '')
    df['Time'] = df['Time'].apply(lambda x: '2017/' + x if len(x) < 8 else x)
    df['Time'] = pd.to_datetime(df['Time'])
    df['Month'] = df['Time'].dt.month
    df['Year'] = df['Time'].dt.year
    # df['Month'] = df['Month'].apply(lambda x: str(x) + '月')
    # df['Month'] = df['Month'].astype('category')
    # 去除Reviewer为#NAME?的记录
    df = df[~df['Reviewer'].str.contains('#NAME?')]
    return df

def get_comparedata(df,merchantslist):
    # 只保留merchantsname的相关记录
    df_current = df[df['Merchant'].isin(merchantslist)]
    # 按餐厅和月汇总2017年的评论数
    df_2017 = df_current[df_current['Year'] == 2017]
    df_2017 = df_2017.groupby(['Month', 'Merchant']).count()['Reviewer']
    # 按照Month列对df_2017排序
    df_2017 = df_2017.sort_index()
    # 去除索引
    df_2017 = df_2017.reset_index()
    df_current['Quarter'] = df_current['Year'].astype(str) + 'Q' + (df_current['Month'] / 3).apply(
        lambda x: math.ceil(x)).astype(str)
    df_2015_2017 = df_current[(df_current['Year'] >= 2015) & (df_current['Year'] <= 2017)]
    df_2015_2017 = df_2015_2017.groupby(['Quarter', 'Merchant']).count()['Reviewer']
    # 按照季度排序
    df_2015_2017 = df_2015_2017.sort_index()
    # 去除索引
    df_2015_2017 = df_2015_2017.reset_index()
    return df_2017,df_2015_2017