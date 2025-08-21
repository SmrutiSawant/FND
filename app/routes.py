from flask import Blueprint, render_template # type: ignore

main = Blueprint("main", __name__)

@main.route("/")
def home():
    return render_template("home.html")

@main.route("/signup")
def signup():
    return render_template("signup.html")

@main.route("/verify")
def verify():
    return render_template("verify.html")
