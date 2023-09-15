import json
import jsonlines
import os

data = {}


def make_talk_data():
    root = "E:/work/AI_Project/scrapy/tutorial/c114通信网爬取数据/对话数据/_talk_html/"
    # file = jsonlines.open("E:/work/AI_Project/scrapy/tutorial/c114通信网爬取数据/对话数据/chatglm_cs_talk_data.json", mode="a")
    file = jsonlines.open("E:/work/AI_Project/scrapy/tutorial/c114通信网爬取数据/对话数据/chatglm_pr_talk_data.json", mode="a")
    path_list = [root + str(i) + "/txt/he/" for i in os.listdir(root)]
    for path in path_list:
        ls = os.listdir(path)
        for id, l in enumerate(ls):
            with open(path + "/" + l, mode='r', encoding='utf-8') as f:
                data = f.readlines()
                prompt = data[0][2: -1].replace('"', "'")
                response = data[1][2: -1].replace('"', "'")
                # one_data = '{"content": "' + prompt + '", ' + '"summary": "' + response + '"}'
                one_data = '{"prompt": "' + prompt + '", ' + '"response": "' + response + '"}'
                temp = one_data
                temp = temp.replace('\\', '\\\\').replace('\n', '').replace('\t', '')
                ct_data = json.loads(str(temp), strict=False)
                file.write(ct_data)
    file.close()


def make_ct_data():
    path = "E:/work/AI_Project/scrapy/tutorial/c114通信网爬取数据/词条数据/all_data/"
    # file = jsonlines.open("E:/work/AI_Project/scrapy/tutorial/c114通信网爬取数据/词条数据/chatglm_cs_ct_data.json", mode="a")
    file = jsonlines.open("E:/work/AI_Project/scrapy/tutorial/c114通信网爬取数据/词条数据/chatglm_pr_ct_data.json", mode="a")
    file_list = [path + str(i) for i in os.listdir(path)]
    for id, fl in enumerate(file_list):
        with open(fl, mode='r', encoding='gb2312', errors="ignore") as f:
            data = f.readlines()
            prompt = data[0][: -1].replace('"', "'")
            response = data[1][: -1].replace('"', "'")
            # one_data = '{"content": "' + prompt + '", ' + '"summary": "' + response + '"}'
            one_data = '{"prompt": "' + prompt + '", ' + '"response": "' + response + '"}'
            temp = one_data
            temp = temp.replace('\\', '\\\\').replace('\n', '').replace('\t', '')
            ct_data = json.loads(str(temp), strict=False)
            file.write(ct_data)
    file.close()


def make_new_data():
    root = r"E:/work/AI_Project/scrapy/tutorial/c114通信网爬取数据/新闻数据/new_data/"
    file = jsonlines.open("E:/work/AI_Project/scrapy/tutorial/c114通信网爬取数据/新闻数据/chatglm_cs_new_data.json", mode="a")
    # file = jsonlines.open("E:/work/AI_Project/scrapy/tutorial/c114通信网爬取数据/新闻数据/chatglm_pr_new_data.json", mode="a")
    file_list = [root + str(i) + "/txt/" for i in os.listdir(root)]
    for fl in file_list:
        ls = os.listdir(fl)
        for id, l in enumerate(ls):
            with open(fl + "/" + l, mode='r', encoding='utf-8') as f:
                data = f.readlines()
                prompt = data[0][: -1].replace('"', "'")
                response = data[1][: -1].replace('"', "'")
                # one_data = '{"prompt": "' + prompt + '", ' + '"response": "' + response + '"}'  # '[{"姓名":"张三","年龄":"18"},{"姓名":"李四","年龄":"20"}]'
                one_data = '{"content": "' + prompt + '", ' + '"summary": "' + response + '"}'  # '[{"姓名":"张三","年龄":"18"},{"姓名":"李四","年龄":"20"}]'
                temp = one_data
                temp = temp.replace('\\', '\\\\').replace('\n', '').replace('\t', '')
                new_data = json.loads(str(temp), strict=False)
                file.write(new_data)
    file.close()


def make_wtxq_data():
    path = "E:/work/AI_Project/NLP/chatGLM/问题小区.xlsx"
    import pandas as pd
    data = pd.read_excel(path)
    print(data.head())
    q = data.iloc[:, 1]
    a = data.iloc[:, 3]
    file = jsonlines.open("E:/work/AI_Project/NLP/chatGLM/chatglm_wtxq_data2.json", mode="a")

    for id in range(q.shape[0]):
        temp = ""
        q_i, a_i = q[id].replace('"', "'"), a[id].replace('"', "'")
        one_data = '{"content": "' + q_i + '", ' + '"summary": "' + a_i + '"}'
        temp = temp + "," + one_data
        # temp = "[" + temp[1:] + "]"
        temp = "" + temp[1:] + ""
        temp = temp.replace('\\', '\\\\').replace('\n', '').replace('\t', '')
        new_data = json.loads(str(temp), strict=False)
        file.write(new_data)
    file.close()

if __name__ == '__main__':
    make_talk_data()
    # make_ct_data()
    # make_new_data()
    # make_wtxq_data()
    # print(data)