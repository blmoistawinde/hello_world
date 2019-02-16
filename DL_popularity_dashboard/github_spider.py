#coding=utf-8
import json
import os
import requests
import datetime
from lxml import etree
import pandas as pd

def get_int_from_xpath(htmlobj,xpath_str):
    text = htmlobj.xpath(xpath_str)[0]
    return int(text.strip().replace(",",""))

if __name__ == "__main__":
    start_urls = [
        'https://github.com/tensorflow/tensorflow',
        'https://github.com/pytorch/pytorch',
        'https://github.com/keras-team/keras',
        'https://github.com/Microsoft/CNTK',
        'https://github.com/apache/incubator-mxnet',
        'https://github.com/PaddlePaddle/Paddle'
    ]
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    to_full = lambda x: os.path.join(THIS_FOLDER, x)
    dt = datetime.datetime.now()
    daily_data_csv = {"date":dt.strftime("%Y-%m-%d")}
    daily_data_json = {"date":dt.strftime("%Y-%m-%d"),"frameworks":{}}
    xpaths = {"stars":'//*[@id="js-repo-pjax-container"]/div[1]/div/ul/li[2]/a[2]/text()',
              "monPRs":'//*[@id="js-repo-pjax-container"]/div[2]/div[1]/div/div[2]/div[2]/ul/li[1]/div[1]/div[2]/span/text()',
              "monIssues":'//*[@id="js-repo-pjax-container"]/div[2]/div[1]/div/div[2]/div[2]/ul/li[1]/div[2]/div[2]/span/text()'}
    
    for url0 in start_urls:
        framework0 = url0.split("/")[-1]
        daily_data_json["frameworks"][framework0] = {}
        result = requests.get(url0+"/pulse/monthly").content
        htmlobj = etree.HTML(result)
        for feature0, xpath_str in xpaths.items():
            val = get_int_from_xpath(htmlobj, xpath_str)
            daily_data_csv["%s_%s" % (framework0, feature0)] = val
            daily_data_json["frameworks"][framework0][feature0] = val
    with open(to_full("daily_record.json"),"w") as f:
        json.dump(daily_data_json,f)

    try:
        # append
        with open(to_full("time_record.csv"),"r",encoding="utf-8") as f:
            df = pd.read_csv(f,index_col=None)
    except:
        # file not exist, init
        df = pd.DataFrame(columns=list(daily_data_csv.keys()))
    df = df.append(daily_data_csv,ignore_index=True)
    df.to_csv(to_full("time_record.csv"),index=None,encoding="utf-8")
    
    
