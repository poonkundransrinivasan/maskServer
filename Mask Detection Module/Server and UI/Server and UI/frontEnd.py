from flask import Flask, render_template
import webbrowser
#from flask_ngrok import run_with_ngrok
import sqlite3 as db
maskDB = db.connect("maskDB2.db")
cur = maskDB.cursor()
app = Flask(__name__)
#def frontEnd():
global data

#run_with_ngrok(app)

headings = ("DATE", "TIME", "ADDRESS", "TEMPRATURE", "REASON", "IMAGE")
#cur.execute("SELECT date,time,address,reason,temp,imageLoc FROM maskData")
#data = cur.fetchall()

img = ".bmp"

@app.route("/home")
@app.route("/")
def home():
    cur.execute("SELECT date,time,address,reason,temp,imageLoc FROM maskData")
    data = cur.fetchall()
    return render_template("home.html", Headings = headings, Data=data, images=img)

@app.route("/AboutUs")
def AboutUs():
    return render_template("AboutUs.html", Headings = headings, Data=data, images=img)

@app.route("/ContactUs")
def ContactUs():
    return render_template("ContactUs.html", Headings = headings, Data=data, images=img)


app.run(host = "0.0.0.0")
