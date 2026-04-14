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
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    
    password = db.Column(db.String(150), nullable=False)

# Create all database tables
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

df = pd.read_csv('groundwater_ml_dataset_cleaned.csv')

@app.route('/')
def home():
    year_min = int(df['year'].min())
    year_max = int(df['year'].max())
    year_range = str(year_min) if year_min == year_max else f"{year_min}–{year_max}"
    stats = {
        'records': len(df),
        'states': df['state'].nunique(),
        'districts': df['district'].nunique(),
        'years': year_range,
        'latest_year': year_max,
        'avg_risk': round(df['risk_score'].mean(), 2),
        'avg_recharge': round(df['annual_recharge'].mean(), 2),
        'avg_extraction': round(df['annual_extraction'].mean(), 2),
        'safe_pct': round(100 * (df['category'] == 'Safe').sum() / len(df), 1),
        'over_pct': round(100 * (df['category'] == 'Over Exploited').sum() / len(df), 1),
        'critical_pct': round(100 * (df['category'] == 'Critical').sum() / len(df), 1),
        'semi_pct': round(100 * (df['category'] == 'Semi Critical').sum() / len(df), 1),
        'top_risk_states': list(df.groupby('state')['risk_score'].mean().nlargest(3).round(1).items()),
    }
    return render_template('home.html', stats=stats)

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

def Stress_Level_Count():
    fig = px.bar(df, x="stress_level",
            title="Stress Level Count")
    graph10_html = pio.to_html(fig, full_html=False)
    return graph10_html

def Risk_Score_Distribution():
    fig = px.box(df, x="state", y="risk_score",
            title="Risk Score Distribution")    
    graph11_html = pio.to_html(fig, full_html=False)
    return graph11_html

def Risk_vs_Extraction_Ratio():
    fig = px.scatter(df, x="extraction_ratio", y="risk_score",
                color="stress_level",
                title="Risk vs Extraction Ratio")
    graph12_html = pio.to_html(fig, full_html=False)
    return graph12_html

def Average_Risk_Over_Time():
    fig = px.line(df.groupby("year")["risk_score"].mean().reset_index(),
            x="year", y="risk_score",
            title="Average Risk Over Time")
    graph13_html = pio.to_html(fig, full_html=False)
    return graph13_html

def  Utilization_Distribution():
    fig = px.violin(df, y="utilization_rate",
                title="Utilization Distribution")
    graph14_html = pio.to_html(fig, full_html=False)
    return graph14_html

def  Utilization_vs_Risk():
    fig = px.scatter(df, x="utilization_rate", y="risk_score",
                color="category",
                title="Utilization vs Risk")
    graph15_html = pio.to_html(fig, full_html=False)
    return graph15_html

def  Stress_vs_Extraction():
    fig = px.box(df, x="stress_level", y="annual_extraction",
            title="Stress vs Extraction")
    graph16_html = pio.to_html(fig, full_html=False)
    return graph16_html

def  Risk_Score_Distribution():
    fig = px.histogram(df, x="risk_score",
                title="Risk Score Distribution")
    graph17_html = pio.to_html(fig, full_html=False)
    return graph17_html

def  Stress_vs_Utilization():
    fig = px.strip(df, x="stress_level", y="utilization_rate",
            title="Stress vs Utilization")
    graph18_html = pio.to_html(fig, full_html=False)
    return graph18_html

def  Hierarchical_View():
    fig = px.sunburst(df, path=["state", "district", "category"],
                values="annual_extraction",
                title="Hierarchical View")
    graph19_html = pio.to_html(fig, full_html=False)
    return graph19_html

def  Resource_Treemap():
    fig = px.treemap(df, path=["state", "district"],
                values="extractable_resource",
                title="Resource Treemap")
    graph20_html = pio.to_html(fig, full_html=False)
    return graph20_html

def  Statewise_Extraction_Trend():
    fig = px.line(df, x="year", y="annual_extraction",
            color="state",
            title="State-wise Extraction Trend")
    graph21_html = pio.to_html(fig, full_html=False)
    return graph21_html

def  State_Comparison():
    fig = px.scatter(df, x="annual_recharge", y="annual_extraction",
                facet_col="state",
                title="State Comparison")
    graph22_html = pio.to_html(fig, full_html=False)
    return graph22_html

def Bubble_Chart():
    fig = px.scatter(df, x="annual_recharge", y="annual_extraction",
                size="extractable_resource",
                color="state",
                title="Bubble Chart")
    graph23_html = pio.to_html(fig, full_html=False)
    return graph23_html

def  Avg_Risk_by_State():
    avg = df.groupby("state")["risk_score"].mean().reset_index()
    fig = px.bar(avg, x="state", y="risk_score",
                title="Avg Risk by State")
    graph24_html = pio.to_html(fig, full_html=False)
    return graph24_html

def  District_Risk_Spread():
    fig = px.box(df, x="district", y="risk_score",
            title="District Risk Spread")
    graph25_html = pio.to_html(fig, full_html=False)
    return graph25_html

def Density_Heatmap():
    fig = px.density_heatmap(df, x="annual_recharge", y="annual_extraction",
                        title="Density Heatmap")
    graph26_html = pio.to_html(fig, full_html=False)
    return graph26_html

def  Category_Distribution_Pie():
    fig = px.pie(df, names="category", values="annual_extraction",
            title="Category Distribution")
    graph27_html = pio.to_html(fig, full_html=False)
    return graph27_html

def  State_wise_Risk_Distribution():
    fig = px.violin(df, x="state", y="risk_score",
                title="State-wise Risk Distribution")
    graph28_html = pio.to_html(fig, full_html=False)
    return graph28_html

#resource & Extraction Analysis pages 
@app.route('/resource_extraction_analysis')
@login_required
def resource_extraction_analysis():
    graph1_html = Annual_Recharge_Trend()
    graph2_html = Statewise_Annual_Recharge()
    graph3_html = Annual_Extraction_Trend()
    graph4_html = Statewise_Annual_Extraction()
    graph5_html = Recharge_vs_Extraction()
    graph6_html = Resource_Distribution()
    graph7_html = Resource_Over_Time()
    graph8_html = Recharge_Distribution()
    graph9_html = Category_Distribution()

    return render_template('resource_extraction_analysis.html', 
                            graph1_html=graph1_html, 
                            graph2_html=graph2_html, 
                            graph3_html=graph3_html, 
                            graph4_html=graph4_html, 
                            graph5_html=graph5_html, 
                            graph6_html=graph6_html, 
                            graph7_html=graph7_html, 
                            graph8_html=graph8_html, 
                            graph9_html=graph9_html)

@app.route('/stress_risk_analysis')
@login_required
def stress_risk_analysis():
    graph10_html = Stress_Level_Count()
    graph11_html = Risk_Score_Distribution()
    graph12_html = Risk_vs_Extraction_Ratio()
    graph13_html = Average_Risk_Over_Time()
    graph14_html = Utilization_Distribution()
    graph15_html = Utilization_vs_Risk()
    graph16_html = Stress_vs_Extraction()
    graph17_html = Risk_Score_Distribution()
    graph18_html = Stress_vs_Utilization()

    return render_template('stress_risk_analysis.html',
                            graph10_html=graph10_html,
                            graph11_html=graph11_html,
                            graph12_html=graph12_html,
                            graph13_html=graph13_html,
                            graph14_html=graph14_html,
                            graph15_html=graph15_html,
                            graph16_html=graph16_html,
                            graph17_html=graph17_html,
                            graph18_html=graph18_html)

@app.route('/state_district_insights')
def state_district_insights():
    graph19_html = Hierarchical_View()
    graph20_html = Resource_Treemap()
    graph21_html = Statewise_Extraction_Trend()
    graph22_html = State_Comparison()
    graph23_html = Bubble_Chart()
    graph24_html = Avg_Risk_by_State()
    graph25_html = District_Risk_Spread()
    graph26_html = Density_Heatmap()
    graph27_html = Category_Distribution_Pie()
    graph28_html = State_wise_Risk_Distribution()
        
    return render_template('state_district_insights.html',
                            graph19_html=graph19_html,
                            graph20_html=graph20_html,
                            graph21_html=graph21_html,
                            graph22_html=graph22_html,
                            graph23_html=graph23_html,
                            graph24_html=graph24_html,
                            graph25_html=graph25_html,
                            graph26_html=graph26_html,
                            graph27_html=graph27_html,
                            graph28_html=graph28_html)


if __name__ == '__main__':
    if not os.path.exists('users.db'):
        with app.app_context():
            db.create_all()
app.run(debug=True)