#!/usr/bin/env python
from flask import Flask, request, render_template
from predict import predict

app = Flask(__name__)

@app.route("/")
def display_form():
    return render_template("main_form.html")

@app.route("/", methods=["POST"])
def predict_author():
    text = request.form["text"]
    result = predict(text)
    return f"Result: {result}"
