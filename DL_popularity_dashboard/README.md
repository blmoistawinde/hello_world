# 深度学习框架热度展板

相关博客：https://blog.csdn.net/blmoistawinde/article/details/87384348

在线演示：http://blmoistawinde.pythonanywhere.com/DL_pop 【每日12:00(北京时间)更新数据】

## 使用
需要安装了`flask`的python3，然后在命令行中：

```bash
flask run
```

就可以在http://127.0.0.1:5000/DL_pop看到对应的展板。

## 主要文件
- app.py: Flask框架建站的主代码
- github_spider.py：使用requests爬取Github上的对应信息
- daily_record.json：日数据
- time_record.json：时间累计数据
- templates/pic_page.html：使用echarts做数据可视化，jquery从Flask后端获得json数据
