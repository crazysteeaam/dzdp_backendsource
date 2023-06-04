import random

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

import src.getdata as getdata
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    df_dish = df_dish.groupby('dish')['dish'].count().sort_values(ascending=False)
    list_dish = []
    for i in range(df_dish.shape[0]):
        dict_dish = {}
        dict_dish['name'] = df_dish.index[i]
        dict_dish['value'] = int(df_dish.iloc[i])
        list_dish.append(dict_dish)

    # 分别获取以上餐厅rating>4,2<rating,rating<2的评论数量
    rating_rate = []
    for i in list(merchants):
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
        "data2": list_dish[:6],
        "data_dish_all": list_dish,
        "data3": rating_rate[:7],
        "comment_level_all": rating_rate,
        "data4": price_list,
        "merchantname": str(merchants[0])
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
        "data1": list_data1,
        "merchantname": merchantname
    }}


@app.get("/userbook")
async def userbook():
    df = getdata.readcsv('static/大众点评评论数据.csv')
    df_reviewer_value = df.groupby('Reviewer')[['Reviewer_value', 'Reviewer_rank']].max()
    level_data = df_reviewer_value['Reviewer_value'].value_counts()
    vip_count = df_reviewer_value['Reviewer_rank'].value_counts()
    df_2017 = df[(df['Year'] == 2017) & (df['Month'] <= 10)]
    df_2017_month = df_2017.groupby('Month')['Reviewer'].count()

    # # 将每个Reviewer去过的餐厅汇总，每个人一行，去重 餐厅间用空格隔开
    # df_reviewer_rest = df.groupby('Reviewer')['Merchant'].unique().apply(lambda x: '<>'.join(x))
    #
    # # 将df_dish转换为data的形式
    # data = []
    # for i in df_reviewer_rest:
    #     # 去掉前后的空格，将中间的空格作为分隔符
    #     i = i.strip().replace('<>', ',')
    #     # 将字符串转换为列表
    #     i = i.split(',')
    #     data.append(i)
    # te = TransactionEncoder()
    # te_ary = te.fit(data).transform(data)
    # df_dish_bool = pd.DataFrame(te_ary, columns=te.columns_)
    # frequent_itemsets = apriori(df_dish_bool, min_support=0.001, use_colnames=True)
    # rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    # 用户聚类
    # 1.将每个用户的Rating、Score_taste、Score_environment、Score_service进行聚合，得到每个用户的评分向量
    df_reviewer = df.groupby('Reviewer')[['Rating', 'Score_taste', 'Score_environment', 'Score_service']].mean()
    # 2.对评分向量进行归一化处理
    scaler = StandardScaler()
    df_reviewer_scaled = scaler.fit_transform(df_reviewer)
    # 3.对用户的评分向量进行聚类
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_reviewer_scaled)
    # 4.降维
    pca = PCA(n_components=2)
    df_reviewer_matrix_2 = pca.fit_transform(df_reviewer_scaled)
    matrix_data = []
    # 5.将每个列表转换为{x:xx,y:xx}的形式
    for i in range(len(df_reviewer_matrix_2)):
        # 取2%的点
        if random.random() < 0.99:
            continue
        else:
            matrix_data.append({'x': round(df_reviewer_matrix_2[i][0], 2), 'y': round(df_reviewer_matrix_2[i][1], 2),
                                'c': int(kmeans.labels_[i])})

    return {"message": {
        "level_data": [
            {"name": "LV." + str(level), "value": int(level_data[level])} for level in level_data.index
        ],
        "vip_data": [
            {"name": "VIP用户" if int(is_vip) else "非VIP用户", "value": int(vip_count[is_vip])} for is_vip in
            vip_count.index
        ],
        "month_data": [
            {"name": str(month) + "月", "value": int(df_2017_month[month])} for month in df_2017_month.index
        ],
        # "rules": rules,
        "matrix_data": matrix_data,
    }}


@app.get("/marketing")
async def marketing_main():
    global advice_rating
    df = getdata.readcsv('static/大众点评评论数据.csv')
    mean_commentnum = df.groupby('Merchant').count().mean()[0]  # 餐厅平均记录数
    ## 获取所有餐厅的人均价格
    df_price = df[['Merchant', 'Price_per_person']]
    # 去掉数据是空这个字的行
    df_price = df_price[~df_price['Price_per_person'].str.contains('空')]
    # 转化为float类型
    df_price['Price_per_person'] = df_price['Price_per_person'].astype(float)
    df_price = df_price.groupby('Merchant')['Price_per_person'].mean().mean()
    vip_rate = round(df.groupby('Reviewer')['Reviewer_rank'].max().mean() * 100, 2)  # 获取VIP占比
    rating_rate = round((df[df['Rating'] >= 4].count() / len(df) * 100)[0], 2)  # 获取Rating >=4 的占比

    # 选择餐厅
    merchants = df['Merchant'].unique()
    merchantsname = merchants[0]

    # 只保留merchantsname的相关记录
    df_current = df[df['Merchant'] == merchantsname]
    commentnum_current = df_current.count()[0]  # 餐厅平均记录数
    ## 获取所有餐厅的人均价格
    df_price_current = df_current[['Merchant', 'Price_per_person']]
    # 去掉数据是空这个字的行
    df_price_current = df_price_current[~df_price_current['Price_per_person'].str.contains('空')]
    # 转化为float类型
    df_price_current['Price_per_person'] = df_price_current['Price_per_person'].astype(float)
    df_price_current = df_price_current.groupby('Merchant')['Price_per_person'].mean().mean()
    vip_rate_current = round(df_current.groupby('Reviewer')['Reviewer_rank'].max().mean() * 100, 2)  # 获取VIP占比
    rating_rate_current = round((df_current[df_current['Rating'] >= 4].count() / len(df_current) * 100)[0],
                                2)  # 获取Rating >=4 的占比

    # 获取当前餐厅每年汇总的评论数
    df_year = df_current.groupby('Year').count()['Merchant']
    # 每行转换为label和value的字典，然后组成的列表
    year_data = []
    for i in range(len(df_year)):
        year_data.append({'label': str(df_year.index[i]) + "年", '评论数量': int(df_year.iloc[i])})

    # 获取当前餐厅2016年每月汇总的评论数
    df_2017 = df_current[df_current['Year'] == 2016]
    df_2017 = df_2017.groupby('Month').count()['Merchant']
    max_month = df_2017.idxmax()
    month_data = []
    for i in range(len(df_2017)):
        month_data.append({'label': str(df_2017.index[i]) + "月", '评论数量': int(df_2017.iloc[i])})

    # 当前餐厅好评中评差评的占比，其中>=4分的为好评，3分为中评，<3分为差评
    df_rating = df_current.groupby('Rating').count()['Merchant']
    # 测出好中差的百分占比
    df_rating = round(df_rating / df_rating.sum() * 100, 2)
    # 每行转换为label和value的字典，然后组成的列表
    rating_label = ['好评', '中评', '差评']
    rating_data = []
    for i in range(len(rating_label)):
        if i == 0:
            rating_data.append({'name': rating_label[i], 'value': round(df_rating[4] + df_rating[5], 2)})
            if round(df_rating[4] + df_rating[5], 2) >= 50:
                advice_rating = '餐厅好评占比高于50％，表现良好。'
            else:
                advice_rating = '餐厅好评占比低于50％，需要提高好评数量来提高餐厅排名。'
        elif i == 1:
            rating_data.append({'name': rating_label[i], 'value': round(df_rating[3], 2)})
        else:
            rating_data.append({'name': rating_label[i], 'value': round(df_rating[1] + df_rating[2], 2)})

    # 菜品热度数据
    df_currentfavourite = df_current['Favorite_foods']
    # 去掉NaN的行
    df_currentfavourite.dropna(inplace=True)

    favour_data = []
    for i in df_currentfavourite:
        # 去掉前后的空格，将中间的空格作为分隔符
        i = i.strip().replace(' ', ',')
        # 将字符串转换为列表
        i = i.split(',')
        for j in i:
            favour_data.append(j)
    # 计数favour_data中每个元素出现的次数
    favour_data = pd.Series(favour_data).value_counts()
    # 列出最热门的五道菜并转换为字符串，用顿号相隔
    high_favour_data = favour_data[:5]
    high_favour_data = '、'.join(high_favour_data.index)
    favourite_data = []
    for i in range(len(favour_data)):
        favourite_data.append({'菜品名称': favour_data.index[i], '热度': int(favour_data.iloc[i])})

    # 获取当前餐厅各rating的平均数
    current_rate = df_current[['Rating', 'Score_service', 'Score_taste', 'Score_environment']].mean()
    current_rate.index = ['总分', '服务', '口味', '环境']
    current_rate = pd.DataFrame(current_rate)
    current_rate.columns = ['当前餐厅']
    all_rate = df[['Rating', 'Score_service', 'Score_taste', 'Score_environment']].mean()
    all_rate.index = ['总分', '服务', '口味', '环境']
    all_rate = pd.DataFrame(all_rate)
    all_rate.columns = ['所有餐厅']
    # 将当前餐厅低于所有餐厅的评分列出
    bad_score = ''
    for i in range(len(current_rate)):
        if current_rate.iloc[i, 0] < all_rate.iloc[i, 0]:
            if i == len(current_rate) - 1:
                bad_score += current_rate.index[i]
            else:
                bad_score += current_rate.index[i] + '、'
    rate_score = pd.merge(current_rate, all_rate, left_index=True, right_index=True)
    rate_data = []
    for i in range(len(rate_score)):
        rate_data.append({'科目': rate_score.index[i], '当前餐厅': round(rate_score.iloc[i, 0], 2),
                          '所有餐厅': round(rate_score.iloc[i, 1], 2), 'fullMark': 5})

    return {
        "message": {
            "data1": [
                {
                    "index": 1,
                    "value": int(commentnum_current),
                    "compare": 1 if commentnum_current > mean_commentnum else 0,
                    "compare_num": round(commentnum_current - mean_commentnum, 0),
                    "name": "累计评论量",
                    "color": "#f1f5fe"
                },
                {
                    "index": 2,
                    "value": "¥" + str(round(df_price_current, 0)),
                    "compare": 1 if df_price_current > df_price else 0,
                    "compare_num": round(df_price_current - df_price, 0),
                    "name": "人均价格",
                    "color": "#f5f0fe"
                },
                {
                    "index": 3,
                    "value": str(round(vip_rate_current, 2)) + "%",
                    "compare": 1 if vip_rate_current > vip_rate else 0,
                    "compare_num": str(round(vip_rate_current - vip_rate, 2)) + "%",
                    "name": "VIP顾客消费占比",
                    "color": "#f1f5fe"
                },
                {
                    "index": 4,
                    "value": str(round(rating_rate_current, 2)) + "%",
                    "compare": 1 if rating_rate_current > rating_rate else 0,
                    "compare_num": str(round(rating_rate_current - rating_rate, 2)) + "%",
                    "name": "好评占比",
                    "color": "#fdf3f1"
                }
            ],
            "merchants": [{"value": merchant, "label": merchant} for merchant in merchants],
            "year_data": year_data,
            "month_data": month_data,
            "rating_data": rating_data,
            "favour_data": favourite_data,
            "rate_data": rate_data,
            "advice": [
                {
                    "index": 1,
                    "content": "餐厅在" + str(max_month) + "月份评论量最多，建议在该月份推出促销活动。",
                },
                {
                    "index": 2,
                    "content": advice_rating,
                },
                {
                    "index": 3,
                    "content": "餐厅" + bad_score + "评分低于全平台整体餐厅得分均值，需要进一步提高该得分情况。",
                },
                {
                    "index": 4,
                    "content": "本餐厅" + high_favour_data + "销量最高，可以考虑未来推出以该菜品为重点打造的套餐。",
                }
            ],
            "merchantname": merchantsname,
        }
    }


@app.get("/marketing/{merchantsname}")
async def marketing(merchantsname: str = None):
    global advice_rating
    df = getdata.readcsv('static/大众点评评论数据.csv')
    mean_commentnum = df.groupby('Merchant').count().mean()[0]  # 餐厅平均记录数
    ## 获取所有餐厅的人均价格
    df_price = df[['Merchant', 'Price_per_person']]
    # 去掉数据是空这个字的行
    df_price = df_price[~df_price['Price_per_person'].str.contains('空')]
    # 转化为float类型
    df_price['Price_per_person'] = df_price['Price_per_person'].astype(float)
    df_price = df_price.groupby('Merchant')['Price_per_person'].mean().mean()
    vip_rate = round(df.groupby('Reviewer')['Reviewer_rank'].max().mean() * 100, 2)  # 获取VIP占比
    rating_rate = round((df[df['Rating'] >= 4].count() / len(df) * 100)[0], 2)  # 获取Rating >=4 的占比

    # 选择餐厅
    merchants = df['Merchant'].unique()

    # 只保留merchantsname的相关记录
    df_current = df[df['Merchant'] == merchantsname]
    commentnum_current = df_current.count()[0]  # 餐厅平均记录数
    ## 获取所有餐厅的人均价格
    df_price_current = df_current[['Merchant', 'Price_per_person']]
    # 去掉数据是空这个字的行
    df_price_current = df_price_current[~df_price_current['Price_per_person'].str.contains('空')]
    # 转化为float类型
    df_price_current['Price_per_person'] = df_price_current['Price_per_person'].astype(float)
    df_price_current = df_price_current.groupby('Merchant')['Price_per_person'].mean().mean()
    vip_rate_current = round(df_current.groupby('Reviewer')['Reviewer_rank'].max().mean() * 100, 2)  # 获取VIP占比
    rating_rate_current = round((df_current[df_current['Rating'] >= 4].count() / len(df_current) * 100)[0],
                                2)  # 获取Rating >=4 的占比

    # 获取当前餐厅每年汇总的评论数
    df_year = df_current.groupby('Year').count()['Merchant']
    # 每行转换为label和value的字典，然后组成的列表
    year_data = []
    for i in range(len(df_year)):
        year_data.append({'label': str(df_year.index[i]) + "年", '评论数量': int(df_year.iloc[i])})

    # 获取当前餐厅2016年每月汇总的评论数
    df_2017 = df_current[df_current['Year'] == 2016]
    df_2017 = df_2017.groupby('Month').count()['Merchant']
    max_month = df_2017.idxmax()
    month_data = []
    for i in range(len(df_2017)):
        month_data.append({'label': str(df_2017.index[i]) + "月", '评论数量': int(df_2017.iloc[i])})

    # 当前餐厅好评中评差评的占比，其中>=4分的为好评，3分为中评，<3分为差评
    df_rating = df_current.groupby('Rating').count()['Merchant']
    # 测出好中差的百分占比
    df_rating = round(df_rating / df_rating.sum() * 100, 2)
    # 每行转换为label和value的字典，然后组成的列表
    rating_label = ['好评', '中评', '差评']
    rating_data = []
    for i in range(len(rating_label)):
        if i == 0:
            rating_data.append({'name': rating_label[i], 'value': round(df_rating[4] + df_rating[5], 2)})
            if round(df_rating[4] + df_rating[5], 2) >= 50:
                advice_rating = '餐厅好评占比高于50％，表现良好。'
            else:
                advice_rating = '餐厅好评占比低于50％，需要提高好评数量来提高餐厅排名。'
        elif i == 1:
            rating_data.append({'name': rating_label[i], 'value': round(df_rating[3], 2)})
        else:
            rating_data.append({'name': rating_label[i], 'value': round(df_rating[1] + df_rating[2], 2)})

    # 菜品热度数据
    df_currentfavourite = df_current['Favorite_foods']
    # 去掉NaN的行
    df_currentfavourite.dropna(inplace=True)

    favour_data = []
    for i in df_currentfavourite:
        # 去掉前后的空格，将中间的空格作为分隔符
        i = i.strip().replace(' ', ',')
        # 将字符串转换为列表
        i = i.split(',')
        for j in i:
            favour_data.append(j)
    # 计数favour_data中每个元素出现的次数
    favour_data = pd.Series(favour_data).value_counts()
    # 列出最热门的五道菜并转换为字符串，用顿号相隔
    high_favour_data = favour_data[:5]
    high_favour_data = '、'.join(high_favour_data.index)
    favourite_data = []
    for i in range(len(favour_data)):
        favourite_data.append({'菜品名称': favour_data.index[i], '热度': int(favour_data.iloc[i])})

    # 获取当前餐厅各rating的平均数
    current_rate = df_current[['Rating', 'Score_service', 'Score_taste', 'Score_environment']].mean()
    current_rate.index = ['总分', '服务', '口味', '环境']
    current_rate = pd.DataFrame(current_rate)
    current_rate.columns = ['当前餐厅']
    all_rate = df[['Rating', 'Score_service', 'Score_taste', 'Score_environment']].mean()
    all_rate.index = ['总分', '服务', '口味', '环境']
    all_rate = pd.DataFrame(all_rate)
    all_rate.columns = ['所有餐厅']
    # 将当前餐厅低于所有餐厅的评分列出
    bad_score = ''
    for i in range(len(current_rate)):
        if current_rate.iloc[i, 0] < all_rate.iloc[i, 0]:
            if i == len(current_rate) - 1:
                bad_score += current_rate.index[i]
            else:
                bad_score += current_rate.index[i] + '、'
    rate_score = pd.merge(current_rate, all_rate, left_index=True, right_index=True)
    rate_data = []
    for i in range(len(rate_score)):
        rate_data.append({'科目': rate_score.index[i], '当前餐厅': round(rate_score.iloc[i, 0], 2),
                          '所有餐厅': round(rate_score.iloc[i, 1], 2), 'fullMark': 5})

    return {
        "message": {
            "data1": [
                {
                    "index": 1,
                    "value": int(commentnum_current),
                    "compare": 1 if commentnum_current > mean_commentnum else 0,
                    "compare_num": round(commentnum_current - mean_commentnum, 0),
                    "name": "累计评论量",
                    "color": "#f1f5fe"
                },
                {
                    "index": 2,
                    "value": "¥" + str(round(df_price_current, 0)),
                    "compare": 1 if df_price_current > df_price else 0,
                    "compare_num": round(df_price_current - df_price, 0),
                    "name": "人均价格",
                    "color": "#f5f0fe"
                },
                {
                    "index": 3,
                    "value": str(round(vip_rate_current, 2)) + "%",
                    "compare": 1 if vip_rate_current > vip_rate else 0,
                    "compare_num": str(round(vip_rate_current - vip_rate, 2)) + "%",
                    "name": "VIP顾客消费占比",
                    "color": "#f1f5fe"
                },
                {
                    "index": 4,
                    "value": str(round(rating_rate_current, 2)) + "%",
                    "compare": 1 if rating_rate_current > rating_rate else 0,
                    "compare_num": str(round(rating_rate_current - rating_rate, 2)) + "%",
                    "name": "好评占比",
                    "color": "#fdf3f1"
                }
            ],
            "merchants": [{"value": merchant, "label": merchant} for merchant in merchants],
            "year_data": year_data,
            "month_data": month_data,
            "rating_data": rating_data,
            "favour_data": favourite_data,
            "rate_data": rate_data,
            "advice": [
                {
                    "index": 1,
                    "content": "餐厅在" + str(max_month) + "月份评论量最多，建议在该月份推出促销活动。",
                },
                {
                    "index": 2,
                    "content": advice_rating,
                },
                {
                    "index": 3,
                    "content": "餐厅" + bad_score + "评分低于全平台整体餐厅得分均值，需要进一步提高该得分情况。",
                },
                {
                    "index": 4,
                    "content": "本餐厅" + high_favour_data + "销量最高，可以考虑未来推出以该菜品为重点打造的套餐。",
                }
            ],
            "merchantname": merchantsname,
        }
    }


@app.get("/compare")
async def compare():
    df = getdata.readcsv('static/大众点评评论数据.csv')
    total_num = df['Merchant'].nunique()
    # 计算各个Merchant的评论数
    df_heat = df['Merchant'].value_counts()
    heat_data = []
    # 前十单独分，其他的归为其他
    for i in range(len(df_heat)):
        if i < 10:
            heat_data.append({'name': df_heat.index[i], 'value': int(df_heat.values[i])})
        elif i == 10:
            heat_data.append({'name': '其他', 'value': int(df_heat.values[i])})
        else:
            heat_data[10]['value'] += df_heat.values[i]
    heat_data[10]['value'] = int(heat_data[10]['value'])
    # 用户复购分析，计算相同餐厅相同用户的评论数
    df_heat2 = df.groupby(['Merchant', 'Reviewer']).size().reset_index().rename(columns={0: 'count'})
    # 相同餐厅相同用户的评论数大于1的为复购用户，计算每家餐厅复购用户和所有用户分别的数量
    df_heat2_all = df_heat2.groupby('Merchant').size().reset_index().rename(columns={0: '总数量'})
    df_heat2 = df_heat2[df_heat2['count'] > 1]
    df_heat2_more = df_heat2.groupby('Merchant').size().reset_index().rename(columns={0: '复购数量'})
    df_heat2 = pd.merge(df_heat2_all, df_heat2_more, on='Merchant')
    df_heat2['复购比例'] = round(df_heat2['复购数量'] / df_heat2['总数量'] * 100, 2)
    # 按复购数量倒序排序
    df_heat2 = df_heat2.sort_values(by='复购数量', ascending=False)
    again_data = []
    for i in range(len(df_heat2)):
        again_data.append({'name': df_heat2['Merchant'].values[i], '获客数': int(df_heat2['总数量'].values[i]),
                           '回头客数量': int(df_heat2['复购数量'].values[i]),
                           '回头客比例': df_heat2['复购比例'].values[i]})
    merchants = df['Merchant'].unique()

    return {
        "message": {
            "heat_data": heat_data,
            "again_data": again_data,
            "again_data_8": again_data[:8],
            "merchants": [{"value": merchant, "label": merchant} for merchant in merchants],
            "total_num": total_num,
            "top1_heat": heat_data[0]['name'],
            "top2_heat": heat_data[1]['name'],
            "top3_heat": heat_data[2]['name'],
            "top1_again": {"name": again_data[0]['name'], "value": again_data[0]['获客数']},
            "top2_again": {"name": again_data[0]['name'], "value": again_data[0]['回头客数量']},
            "top3_again": {"name": again_data[0]['name'], "value": again_data[0]['回头客比例']},
            "top4_again": again_data[1]['name'],
            "top5_again": again_data[2]['name'],
            "top6_again": again_data[3]['name'],
            "top7_again": again_data[4]['name'],
        }
    }


@app.get("/marketshare")
async def marketshare():
    df = getdata.readcsv('static/大众点评评论数据.csv')
    merchants = df['Merchant'].unique()
    # 计算各个Merchant的评论数
    df_heat = df['Merchant'].value_counts()
    heat_data = []
    total_heat = 0
    for i in range(len(df_heat)):
        heat_data.append({'name': df_heat.index[i], 'value': int(df_heat.values[i])})
        total_heat += int(df_heat.values[i])
    return {
        "message": {
            "heat_data": heat_data,
            "total_heat": total_heat,
            "merchants": [{"value": merchant, "label": merchant} for merchant in merchants],
        }
    }


@app.get("/compare3/{merchant1}/{merchant2}/{merchant3}")
async def compare3(merchant1, merchant2, merchant3):
    df = getdata.readcsv('static/大众点评评论数据.csv')
    merchantslist = [merchant1, merchant2, merchant3]
    df_2017, df_2015_2017 = getdata.get_comparedata(df, merchantslist)
    month_data = []
    # 将每月不同餐厅的评论数放入month_data
    for i in range(len(df_2017['Month'].unique())):
        month_data.append(
            {'name': str(df_2017['Month'].unique()[i]) + "月",
             'merchant1': int(df_2017[(df_2017['Month'] == df_2017['Month'].unique()[i]) & (
                     df_2017['Merchant'] == merchantslist[0])]['Reviewer'].values if df_2017[
                 (df_2017['Month'] == df_2017['Month'].unique()[i]) & (df_2017['Merchant'] == merchantslist[0])][
                 'Reviewer'].values else 0),
             'merchant2': int(df_2017[(df_2017['Month'] == df_2017['Month'].unique()[i]) & (
                     df_2017['Merchant'] == merchantslist[1])]['Reviewer'].values if df_2017[
                 (df_2017['Month'] == df_2017['Month'].unique()[i]) & (df_2017['Merchant'] == merchantslist[1])][
                 'Reviewer'].values else 0),
             'merchant3': int(df_2017[(df_2017['Month'] == df_2017['Month'].unique()[i]) & (
                     df_2017['Merchant'] == merchantslist[2])]['Reviewer'].values if df_2017[
                 (df_2017['Month'] == df_2017['Month'].unique()[i]) & (df_2017['Merchant'] == merchantslist[2])][
                 'Reviewer'].values else 0)
             })
    quarter_data = []
    for i in range(len(df_2015_2017['Quarter'].unique())):
        quarter_data.append(
            {'name': str(df_2015_2017['Quarter'].unique()[i]),
             'merchant1': int(df_2015_2017[(df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                     df_2015_2017['Merchant'] == merchantslist[0])]['Reviewer'].values if df_2015_2017[
                 (df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                         df_2015_2017['Merchant'] == merchantslist[0])]['Reviewer'].values else 0),
             'merchant2': int(df_2015_2017[(df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                     df_2015_2017['Merchant'] == merchantslist[1])]['Reviewer'].values if df_2015_2017[
                 (df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                         df_2015_2017['Merchant'] == merchantslist[1])]['Reviewer'].values else 0),
             'merchant3': int(df_2015_2017[(df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                     df_2015_2017['Merchant'] == merchantslist[2])]['Reviewer'].values if df_2015_2017[
                 (df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                         df_2015_2017['Merchant'] == merchantslist[2])]['Reviewer'].values else 0)
             })
    return {
        "message": {
            "month_data": month_data,
            "quarter_data": quarter_data
        }
    }


@app.get("/compare2/{merchant1}/{merchant2}")
async def compare2(merchant1, merchant2):
    df = getdata.readcsv('static/大众点评评论数据.csv')
    merchantslist = [merchant1, merchant2]
    df_2017, df_2015_2017 = getdata.get_comparedata(df, merchantslist)
    month_data = []
    # 将每月不同餐厅的评论数放入month_data
    for i in range(len(df_2017['Month'].unique())):
        month_data.append(
            {'name': str(df_2017['Month'].unique()[i]) + "月",
             'merchant1': int(df_2017[(df_2017['Month'] == df_2017['Month'].unique()[i]) & (
                     df_2017['Merchant'] == merchantslist[0])]['Reviewer'].values if df_2017[
                 (df_2017['Month'] == df_2017['Month'].unique()[i]) & (df_2017['Merchant'] == merchantslist[0])][
                 'Reviewer'].values else 0),
             'merchant2': int(df_2017[(df_2017['Month'] == df_2017['Month'].unique()[i]) & (
                     df_2017['Merchant'] == merchantslist[1])]['Reviewer'].values if df_2017[
                 (df_2017['Month'] == df_2017['Month'].unique()[i]) & (df_2017['Merchant'] == merchantslist[1])][
                 'Reviewer'].values else 0)
             })
    quarter_data = []
    for i in range(len(df_2015_2017['Quarter'].unique())):
        quarter_data.append(
            {'name': str(df_2015_2017['Quarter'].unique()[i]),
             'merchant1': int(df_2015_2017[(df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                     df_2015_2017['Merchant'] == merchantslist[0])]['Reviewer'].values if df_2015_2017[
                 (df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                         df_2015_2017['Merchant'] == merchantslist[0])]['Reviewer'].values else 0),
             'merchant2': int(df_2015_2017[(df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                     df_2015_2017['Merchant'] == merchantslist[1])]['Reviewer'].values if df_2015_2017[
                 (df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                         df_2015_2017['Merchant'] == merchantslist[1])]['Reviewer'].values else 0)
             })
    return {
        "message": {
            "month_data": month_data,
            "quarter_data": quarter_data
        }
    }


@app.get("/compare1/{merchant1}")
async def compare1(merchant1):
    df = getdata.readcsv('static/大众点评评论数据.csv')
    merchantslist = [merchant1]
    df_2017, df_2015_2017 = getdata.get_comparedata(df, merchantslist)
    month_data = []
    # 将每月不同餐厅的评论数放入month_data
    for i in range(len(df_2017['Month'].unique())):
        month_data.append(
            {'name': str(df_2017['Month'].unique()[i]) + "月",
             'merchant1': int(df_2017[(df_2017['Month'] == df_2017['Month'].unique()[i]) & (
                     df_2017['Merchant'] == merchantslist[0])]['Reviewer'].values if df_2017[
                 (df_2017['Month'] == df_2017['Month'].unique()[i]) & (df_2017['Merchant'] == merchantslist[0])][
                 'Reviewer'].values else 0)
             })
    quarter_data = []
    for i in range(len(df_2015_2017['Quarter'].unique())):
        quarter_data.append(
            {'name': str(df_2015_2017['Quarter'].unique()[i]),
             'merchant1': int(df_2015_2017[(df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                     df_2015_2017['Merchant'] == merchantslist[0])]['Reviewer'].values if df_2015_2017[
                 (df_2015_2017['Quarter'] == df_2015_2017['Quarter'].unique()[i]) & (
                         df_2015_2017['Merchant'] == merchantslist[0])]['Reviewer'].values else 0)
             })
    return {
        "message": {
            "month_data": month_data,
            "quarter_data": quarter_data
        }
    }


@app.get("/raredata")
async def raredata():
    df = getdata.readcsv('static/大众点评评论数据.csv')
    # 将每行存为一个json，放入列表中
    columns = ['Review_ID', 'Merchant', 'Rating', 'Score_taste', 'Score_environment',
               'Score_service', 'Price_per_person', 'Time', 'Num_thumbs_up',
               'Num_ response', 'Content_review', 'Reviewer', 'Reviewer_value',
               'Reviewer_rank', 'Favorite_foods']
    rare_data = []
    for i in df.index:
        if i == 40000:
            break
        rare_data.append(
            {'Review_ID': str(df.loc[i, 'Review_ID']),
             'Merchant': str(df.loc[i, 'Merchant']),
             'Rating': str(df.loc[i, 'Rating']),
             'Score_taste': str(df.loc[i, 'Score_taste']),
             'Score_environment': str(df.loc[i, 'Score_environment']),
             'Score_service': str(df.loc[i, 'Score_service']),
             'Price_per_person': str(df.loc[i, 'Price_per_person']),
             'Time': str(df.loc[i, 'Time']),
             'Num_thumbs_up': str(df.loc[i, 'Num_thumbs_up']),
             'Num_ response': str(df.loc[i, 'Num_ response']),
             'Content_review': str(df.loc[i, 'Content_review']),
             'Reviewer': str(df.loc[i, 'Reviewer']),
             'Reviewer_value': str(df.loc[i, 'Reviewer_value']),
             'Reviewer_rank': str(df.loc[i, 'Reviewer_rank']),
             'Favorite_foods': str(df.loc[i, 'Favorite_foods'])
             })

    return {
        "message": {
            "raredata": rare_data
        }
    }


if __name__ == "__main__":
    # uvicorn.run(app="main:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(app="main:app", host="0.0.0.0", port=8008, reload=True, ssl_keyfile="0000.key", ssl_certfile="0000.crt")
