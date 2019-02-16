# coding=utf-8
from flask import Flask, redirect, render_template, request, url_for, jsonify

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route("/", methods=["GET"])
def index():
    return render_template("main_page.html")


@app.route("/DL_pop", methods=["GET"])
def DL_pop():
    return render_template("pic_page.html")


@app.route("/gitdl_api", methods=["GET"])
def gitdl_api():
    import json
    import os
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    data = {"date":"","frameworks":["tensorflow", "keras", "pytorch", "mxnet", "CNTK", "Paddle"]}
    data["stars"] = [0 for f in data["frameworks"]]
    data["develop"] = [0.0 for f in data["frameworks"]]
    data["use"] = [0.0 for f in data["frameworks"]]
    data["pop"] = []
    with open(os.path.join(THIS_FOLDER, 'daily_record.json'),encoding="utf-8") as f:
        record = json.load(f)
    data["date"] = record["date"]
    
    for framework0, feats in record["frameworks"].items():
        framework0 = "mxnet" if framework0 == "incubator-mxnet" else framework0
        id0 = data["frameworks"].index(framework0)
        stars, monPRs, monIssues = feats["stars"], feats["monPRs"], feats["monIssues"]
        data["stars"][id0] = stars
        data["develop"][id0] = float(1000*monPRs/stars)
        data["use"][id0] = float(1000*monIssues / stars)
        data["pop"].append({"name":framework0,"value":monIssues})
    return jsonify(data)

if __name__ == "__main__":
    app.run()
