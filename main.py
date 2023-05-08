from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import src.getdata as getdata
import pandas as pd

import uvicorn

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/analyze")
async def analyze_main():
    df = getdata.readcsv('static/大众点评评论数据.csv')
    total_num = df['Merchant'].nunique()
    total_comment = df.shape[0]
    mean_rating = round(df['Rating'].mean(), 2)
    vip_rate = round(df.groupby('Reviewer')['Reviewer_rank'].max().mean() * 100, 2)
    merchants = df['Merchant'].unique()

    # 求第1家merchants 按月汇总2017年1月-2017年10月每月的平均评分并去掉空行
    df_data1 = df[(df['Year'] == 2017)].groupby('Month')[
        ['Score_taste', 'Score_environment', 'Score_service']].mean().dropna()
    df_data1.reset_index(inplace=True)
    list_data1 = []
    for i in range(df_data1.shape[0]):
        dict_data1 = {}
        dict_data1['name'] = str(df_data1.iloc[i, 0]) + '月'
        dict_data1['味道评分'] = round(df_data1.iloc[i, 1], 2)
        dict_data1['环境评分'] = round(df_data1.iloc[i, 2], 2)
        dict_data1['服务评分'] = round(df_data1.iloc[i, 3], 2)
        list_data1.append(dict_data1)

    # 获取最热门的6道菜
    df_dish = df['Favorite_foods']
    df_dish.dropna(inplace=True)
    # 将df_dish转换为data的形式
    data = []
    for i in df_dish:
        # 去掉前后的空格，将中间的空格作为分隔符
        i = i.strip().replace(' ', ',')
        # 将字符串转换为列表
        i = i.split(',')
        for j in i:
            data.append(j)
    # 汇总计算每道菜出现的次数，并取前6道
    df_dish = pd.DataFrame(data, columns=['dish'])
    df_dish = df_dish.groupby('dish')['dish'].count().sort_values(ascending=False).head(6)
    list_dish = []
    for i in range(df_dish.shape[0]):
        dict_dish = {}
        dict_dish['name'] = df_dish.index[i]
        dict_dish['value'] = int(df_dish.iloc[i])
        list_dish.append(dict_dish)

    # 分别获取以上餐厅rating>4,2<rating,rating<2的评论数量
    rating_rate = []
    for i in list(merchants[:7]):
        rate_dict = {}
        rate_dict['name'] = i
        rate_dict['好评'] = df[(df['Merchant'] == i) & (df['Rating'] >= 4)].shape[0]
        rate_dict['中评'] = df[(df['Merchant'] == i) & (df['Rating'] >= 2) & (df['Rating'] < 4)].shape[0]
        rate_dict['差评'] = df[(df['Merchant'] == i) & (df['Rating'] < 2)].shape[0]
        rating_rate.append(rate_dict)

    ## 获取所有餐厅的人均价格
    df_price = df[['Merchant', 'Price_per_person']]
    # 去掉数据是空这个字的行
    df_price = df_price[~df_price['Price_per_person'].str.contains('空')]
    # 转化为float类型
    df_price['Price_per_person'] = df_price['Price_per_person'].astype(float)
    df_price = df_price.groupby('Merchant')['Price_per_person'].mean()

    # 分类计数人均50以内，50-100，100-150，150-200，200-500，500-1000，1000以上的餐厅数量
    df_price1 = df_price[df_price <= 50].count()
    df_price2 = df_price[(df_price > 50) & (df_price <= 100)].count()
    df_price3 = df_price[(df_price > 100) & (df_price <= 150)].count()
    df_price4 = df_price[(df_price > 150) & (df_price <= 200)].count()
    df_price5 = df_price[(df_price > 200) & (df_price <= 500)].count()
    df_price6 = df_price[(df_price > 500) & (df_price <= 1000)].count()
    df_price7 = df_price[df_price > 1000].count()
    df_price_count = [df_price1, df_price2, df_price3, df_price4, df_price5, df_price6, df_price7]
    df_price_name = ['50元以下', '50-100元', '100-150元', '150-200元', '200-500元', '500-1000元', '1000元以上']
    # 汇总为{"name": "50元以下","value": 30}的形式
    price_list = []
    for i in range(len(df_price_count)):
        price_dict = {}
        price_dict['name'] = df_price_name[i]
        price_dict['value'] = int(df_price_count[i])
        price_list.append(price_dict)

    return {"message": {
        "total_num": total_num,
        "total_comment": total_comment,
        "mean_rating": mean_rating,
        "vip_rate": vip_rate,
        "merchants": [{"value": merchant, "label": merchant} for merchant in merchants],
        "data1": list_data1,
        "data2": list_dish,
        "data3": rating_rate,
        "data4": price_list
    }}


@app.get("/analyze_data1/{merchantname}")
async def analyze_data1(merchantname: str):
    print(merchantname)
    df = getdata.readcsv('static/大众点评评论数据.csv')
    # 求第1家merchants 按月汇总2017年1月-2017年10月每月的平均评分并去掉空行
    df_data1 = df[(df['Year'] == 2017) & (df['Merchant'] == merchantname)].groupby('Month')[
        ['Score_taste', 'Score_environment', 'Score_service']].mean().dropna()
    df_data1.reset_index(inplace=True)
    list_data1 = []
    for i in range(df_data1.shape[0]):
        dict_data1 = {}
        dict_data1['name'] = str(df_data1.iloc[i, 0]) + '月'
        dict_data1['味道评分'] = round(df_data1.iloc[i, 1], 2)
        dict_data1['环境评分'] = round(df_data1.iloc[i, 2], 2)
        dict_data1['服务评分'] = round(df_data1.iloc[i, 3], 2)
        list_data1.append(dict_data1)

    return {"message": {
        "data1": list_data1
    }}


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=8008, reload=True,ssl_keyfile="0000.key", ssl_certfile="0000.crt")
