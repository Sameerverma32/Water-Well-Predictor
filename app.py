from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import bcrypt



app = Flask(__name__)
app.config['SECRET KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    
    password = db.Column(db.String(150), nullable=False)

# Create all database tables
with app.app_context():
    db.create_all()

df = pd.read_csv('groundwater_ml_dataset_cleaned.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

#graphs functions
def Annual_Recharge_Trend():
    fig = px.line(df, x="year", y="annual_recharge", color="state",
            title="Annual Recharge Trend")
    graph1_html = pio.to_html(fig, full_html=False)
    return graph1_html

def Statewise_Annual_Recharge():
    fig = px.bar(df, x="state", y="annual_recharge",
            title="State-wise Annual Recharge")
    graph2_html = pio.to_html(fig, full_html=False)
    return graph2_html

def Annual_Extraction_Trend():
    fig = px.line(df, x="year", y="annual_extraction", color="state",
            title="Annual Extraction Trend")
    graph3_html = pio.to_html(fig, full_html=False)
    return graph3_html

def Statewise_Annual_Extraction():
    fig = px.bar(df, x="state", y="annual_extraction",
            title="State-wise Annual Extraction")   
    graph4_html = pio.to_html(fig, full_html=False)
    return graph4_html

def Recharge_vs_Extraction():
    fig = px.scatter(df, x="annual_recharge", y="annual_extraction",
                color="state",
                title="Recharge vs Extraction")
    graph5_html = pio.to_html(fig, full_html=False)
    return graph5_html

def Resource_Distribution():
    fig = px.box(df, x="state", y="extractable_resource",
            title="Resource Distribution")
    graph6_html = pio.to_html(fig, full_html=False)
    return graph6_html

def Resource_Over_Time():
    fig = px.area(df, x="year", y="extractable_resource",
            title="Resource Over Time")
    graph7_html = pio.to_html(fig, full_html=False)
    return graph7_html

def Recharge_Distribution():
    fig = px.histogram(df, x="annual_recharge",
                title="Recharge Distribution")
    graph8_html = pio.to_html(fig, full_html=False)
    return graph8_html

def Category_Distribution():
    fig = px.pie(df, names="category",
            title="Category Distribution")
    graph9_html = pio.to_html(fig, full_html=False)
    return graph9_html







if __name__ == '__main__':
    if not os.path.exists('users.db'):
        with app.app_context():
            db.create_all()
app.run(debug=True)