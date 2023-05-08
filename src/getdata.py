import pandas as pd


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
    # 清洗Time中不含有年份的数据
    df['Time'] = df['Time'].str.replace('月', '/').str.replace('日', '')
    df['Time'] = df['Time'].apply(lambda x: '2017/' + x if len(x) < 8 else x)
    df['Time'] = pd.to_datetime(df['Time'])
    df['Month'] = df['Time'].dt.month
    df['Year'] = df['Time'].dt.year
    # df['Month'] = df['Month'].apply(lambda x: str(x) + '月')
    # df['Month'] = df['Month'].astype('category')
    return df