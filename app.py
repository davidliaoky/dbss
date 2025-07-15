from flask import Flask, render_template, request
import joblib
from groq import Groq
from openai import OpenAI

# Need to add K.e.y. here
import os
#for cloud
os.environ['GROQ_API_KEY'] = os.getenv("groq")

# must correspond to file name
app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/main",methods=["GET","POST"])
def main():
    q = request.form.get("q")
    #db
    return(render_template("main.html"))

@app.route("/llama",methods=["GET","POST"])
def llama():
    return(render_template("llama.html"))

@app.route("/llama_reply",methods=["GET","POST"])
def llama_reply():
    q = request.form.get("q")
    # load model

    client = Groq()
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return(render_template("llama_reply.html",r=completion.choices[0].message.content))

@app.route("/sealion",methods=["GET","POST"])
def sealion():
    return(render_template("sealion.html"))

@app.route("/sealion_reply",methods=["GET","POST"])
def sealion_reply():
    q = request.form.get("q")
    # load model

    client = OpenAI(
        #api_key = "sk-7GVcy5oGCRXOrEW3qh0Z1w",
        api_key=os.getenv("sealion"),
        base_url="https://api.sea-lion.ai/v1"
    )

    completion = client.chat.completions.create(
        model="aisingapore/Gemma-SEA-LION-v3-9B-IT",
        messages=[
            {
            "role": "user",
            "content": "what is kaypoh ji"
            }
        ]
    )
    return(render_template("llama_reply.html",r=completion.choices[0].message.content))

@app.route("/dbs",methods=["GET","POST"])
def dbs():
    return(render_template("dbs.html"))

@app.route("/prediction",methods=["GET","POST"])
def prediction():
    q = float(request.form.get("q"))
    #return(render_template("prediction.html",r=(-50.6*q)+90.2))

    # load model
    model = joblib.load("dbs.jl")

    # make prediction
    pred = model.predict([[q]])

    return(render_template("prediction.html",r=pred))

# for local testing
if __name__ == "__main__":
    app.run()