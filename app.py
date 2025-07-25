from flask import Flask, render_template, request
import joblib
from groq import Groq
from openai import OpenAI
import requests
import sqlite3
from datetime import datetime

# Need to add K.e.y. here
import os
#for cloud
os.environ['GROQ_API_KEY'] = os.getenv("groq")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# must correspond to file name
app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@app.route("/main",methods=["GET","POST"])
def main():
    name = request.form.get("q")
    #db - insert
    # Get current timestamp
    now = datetime.now()

    # Insert into SQLite DB
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()

    # Insert user input and timestamp
    cursor.execute("INSERT INTO user (name, timestamp) VALUES (?, ?)", (name, now))
    conn.commit()
    conn.close()
    
    return render_template("main.html", name=name, timestamp=now)

@app.route("/llama",methods=["GET","POST"])
def llama():
    return(render_template("llama.html"))

@app.route("/llama_reply",methods=["GET","POST"])
def llama_reply():
    q = request.form.get("q")
    # load model

    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    completion1 = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": q
            }
        ]
    )
    return(render_template("llama_reply.html",r=completion.choices[0].message.content, r1=completion1.choices[0].message.content))

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
            "content": q
            }
        ]
    )
    return(render_template("sealion_reply.html",r=completion.choices[0].message.content))

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

@app.route("/telegram",methods=["GET","POST"])
def telegram():

    domain_url = 'https://dbss-1-5r5x.onrender.com'

    # The following line is used to delete the existing webhook URL for the Telegram bot
    delete_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    requests.post(delete_webhook_url, json={"url": domain_url, "drop_pending_updates": True})
    
    # Set the webhook URL for the Telegram bot
    set_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook?url={domain_url}/webhook"
    webhook_response = requests.post(set_webhook_url, json={"url": domain_url, "drop_pending_updates": True})

    if webhook_response.status_code == 200:
        # set status message
        status = "The telegram bot is running. Please check with the telegram bot. @dsai_lky_bot"
    else:
        status = "Failed to start the telegram bot. Please check the logs."
    
    return(render_template("telegram.html", status=status))

@app.route("/stop_telegram",methods=["GET","POST"])
def stop_telegram():

    domain_url = 'https://dbss-1-5r5x.onrender.com'

    # The following line is used to delete the existing webhook URL for the Telegram bot
    delete_webhook_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook"
    requests.post(delete_webhook_url, json={"url": domain_url, "drop_pending_updates": True})
    webhook_response = requests.post(delete_webhook_url, json={"url": domain_url, "drop_pending_updates": True})
    
    if webhook_response.status_code == 200:
        # set status message
        status = "The telegram bot has stopped."
    else:
        status = "Failed to stop the telegram bot. Please check the logs."
    
    return(render_template("telegram.html", status=status))

@app.route("/webhook",methods=["GET","POST"])
def webhook():

    # This endpoint will be called by Telegram when a new message is received
    update = request.get_json()
    if "message" in update and "text" in update["message"]:
        # Extract the chat ID and message text from the update
        chat_id = update["message"]["chat"]["id"]
        query = update["message"]["text"]

        # Pass the query to the Groq model
        client = Groq()
        completion_ds = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        response_message = completion_ds.choices[0].message.content

        # Send the response back to the Telegram chat
        send_message_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(send_message_url, json={
            "chat_id": chat_id,
            "text": response_message
        })
    return('ok', 200)
    
@app.route("/user_log", methods=["GET"])
def user_log():
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT rowid, name, timestamp FROM user ORDER BY timestamp DESC")
    records = cursor.fetchall()

    conn.close()
    
    return render_template("user_log.html", records=records)

@app.route("/delete_log", methods=["POST"])
def delete_log():
    rowid = request.form.get("rowid")

    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM user WHERE rowid = ?", (rowid,))
    conn.commit()
    conn.close()

    return render_template("delete_log.html", rowid=rowid)

@app.route("/delete_all_logs", methods=["POST"])
def delete_all_logs():
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user")
    conn.commit()
    conn.close()
    return render_template("delete_log.html", rowid="ALL")

@app.route("/emotion", methods=["GET"])
def emotion():
    return render_template("emotion.html")

# for local testing
if __name__ == "__main__":
    app.run()