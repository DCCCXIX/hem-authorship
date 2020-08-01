from flask import Flask, request, render_template

app = Flask(__name__)


@app.route("/")
def display_form():
    return render_template("main_form.html")


@app.route("/", methods=["POST"])
def my_form_post():
    text = request.form["text"]
    processed_text = text.upper()
    return f"this is your uppercased text: {processed_text}"
