import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import streamlit as st
from tensorflow.keras.models import load_model

# Load your data
final_df = pd.read_csv(r"C:\Users\khang\Elite 14 M-L\Projects\IPL-Data-set\final_ipl_data1.csv")
teams_data = pd.read_csv(r"C:\Users\khang\Elite 14 M-L\Projects\IPL-Data-set\teams_data.csv")
final_match = pd.read_csv(r"C:\Users\khang\Elite 14 M-L\Projects\IPL-Data-set\final_match.csv")
batsmen_data = pd.read_csv(r"C:\Users\khang\Elite 14 M-L\Projects\IPL-Data-set\batsmen_data.csv")
bowlers_data = pd.read_csv(r"C:\Users\khang\Elite 14 M-L\Projects\IPL-Data-set\bowlers_data.csv")

# Splitting the data
fv = final_df.iloc[:, :-1]
cv = final_df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(fv, cv, test_size=0.20, random_state=3, stratify=cv)

# Dividing the dataset based on the type of the variables
numerical_data = x_train.select_dtypes(include=["int64", "float64"])
cat_data = x_train.select_dtypes(include=["object"])

# Pipeline to impute and encode nominal columns
num_p = Pipeline([("imputing_n", SimpleImputer()), ("scaling", StandardScaler())])
cp = Pipeline([("imputing_c", SimpleImputer(strategy="most_frequent")), ("Encoder", OneHotEncoder())])

# Pipeline for column transformation to apply for different types of data
ct = ColumnTransformer([("nominal", cp, cat_data.columns), ("numerical", num_p, numerical_data.columns)], remainder="passthrough")

model = Pipeline([("ct", ct), ("algo", LogisticRegression(C=10, penalty='l1', solver='liblinear'))])
model.fit(x_train, y_train)

def main():
    # Add navigation sidebar
    page = st.sidebar.selectbox("Select Page", ["Home Page", "Batsman Stats", "Bowler Stats", "IPL_Winner_Prediction_Probability(LR)", "IPL Winning Prediction Using ANN", "Trophy Winners and Runner Ups", "Team History"])
    
    # Display the image and content based on the selected page
    if page == "Home Page":
        st.title("IPL Dashboard")
        st.image(r"C:\Users\khang\Elite 14 M-L\Projects\IPL-Data-set\ipl image.jpg", caption='IPL Image')
        
    elif page == "Batsman Stats":
        st.header("Batsman Stats")
        selected_batsman = st.selectbox("Select Batsman", batsmen_data['batsman'].unique(), key="batsman_select")

        if st.button("Submit Batsman Stats"):
            a = batsmen_data[batsmen_data['batsman'] == selected_batsman]
            st.write("Total Runs:", a['total_runs'].sum())

        if st.button("Submit Batsman Stats Details"):
            a = batsmen_data[batsmen_data['batsman'] == selected_batsman]
            st.write("No. of half-centuries:", len(a[(a["total_runs"] >= 50) & (a["total_runs"] < 100)]))
            st.write("No. of centuries:", len(a[(a["total_runs"] >= 100) & (a["total_runs"] < 200)]))
            st.write("No. of double centuries:", len(a[(a["total_runs"] >= 200) & (a["total_runs"] < 300)]))
            st.write("Total runs in IPL:", a["total_runs"].sum())
            plt.figure(figsize=(10, 6))
            plt.plot(a['Season'], a['total_runs'], marker='o')
            plt.xlabel('Season')
            plt.ylabel('Runs')
            plt.title(f'Runs Across Seasons for {selected_batsman}')
            plt.xticks(rotation=45)
            st.pyplot(plt)

    elif page == "Bowler Stats":
        st.header("Bowler Stats")
        selected_bowler = st.selectbox("Select Bowler", bowlers_data['bowler'].unique(), key="bowler_select")

        if st.button("Submit Bowler Stats"):
            a = bowlers_data[bowlers_data["bowler"] == selected_bowler]
            st.write("Total Wickets:", a["total_wickets"].iloc[0])
            # No need to sum the total wickets again
            # st.write("No. of wickets:", a["total_wickets"].sum())
            # Check if the DataFrame contains the "Season" column before plotting
            if "Season" in a.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(a.index, a['total_wickets'], marker='o')  # Assuming index represents different seasons
                plt.xlabel('Season')
                plt.ylabel('Wickets')
                plt.title(f'Wickets Across Seasons by {selected_bowler}')
                plt.xticks(rotation=45)
                st.pyplot(plt)
           

    elif page == "IPL_Winner_Prediction_Probability(LR)":
        st.title('IPL Winning Prediction')
        teams = ['----Select----',
                 'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
                 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore',
                 'Sunrisers Hyderabad']
        col1, col2 = st.columns(2)
        with col1:
            batting_team = st.selectbox('Select Batting Team', teams)
        with col2:
            if batting_team == '----Select----':
                bowling_team = st.selectbox('Select Bowling Team', teams)
            else:
                filtered_teams = [team for team in teams if team != batting_team]
                bowling_team = st.selectbox('Select Bowling Team', filtered_teams)
        target = st.number_input('Target')
        col1, col2, col3 = st.columns(3)
        with col1:
            score = st.number_input('Score', step=1, format="%d", value=0)
        with col2:
            overs = st.number_input("Over Completed", step=0.1, min_value=0.0, max_value=20.0)
        with col3:
            wickets = st.number_input("Wickets down", step=1, format="%d", value=0, min_value=0, max_value=10)
        if st.button('Predict Winning Probability'):
            runs_left = target - score
            balls_left = 120 - (overs * 6)
            wickets = 10 - wickets
            crr = score / overs
            rrr = runs_left / (balls_left / 6)
            input_data = pd.DataFrame({'BattingTeam': [batting_team], 'BowlingTeam': [bowling_team],
                                       'runs_left': [runs_left], 'balls_left': [balls_left],
                                       'wickets_remaining': [wickets], 'target': [target], 'crr': [crr], 'rrr': [rrr]})
            result = model.predict_proba(input_data)
            loss = result[0][0]
            win = result[0][1]
            st.header(batting_team + " = " + str(round(win * 100)) + "%")
            st.header(bowling_team + " = " + str(round(loss * 100)) + "%")

    elif page == "IPL Winning Prediction Using ANN":
        st.title('IPL Winning Prediction Using ANN')
        best_model = load_model(r"C:\Users\khang\Elite 14 M-L\Projects\IPL-Data-set\best_ipl_model_weights.keras")
        teams = ['----Select----',
                 'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
                 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore',
                 'Sunrisers Hyderabad']
        col1, col2 = st.columns(2)
        with col1:
            batting_team = st.selectbox('Select Batting Team', teams)
        with col2:
            if batting_team == '----Select----':
                bowling_team = st.selectbox('Select Bowling Team', teams)
            else:
                filtered_teams = [team for team in teams if team != batting_team]
                bowling_team = st.selectbox('Select Bowling Team', filtered_teams)
        target = st.number_input('Target')
        col1, col2, col3 = st.columns(3)
        with col1:
            score = st.number_input('Score', step=1, format="%d", value=0)
        with col2:
            overs = st.number_input("Over Completed", step=0.1, min_value=0.0, max_value=20.0)
        with col3:
            wickets = st.number_input("Wickets down", step=1, format="%d", value=0, min_value=0, max_value=10)
        if st.button('Predict Winning Probability'):
            runs_left = target - score
            balls_left = 120 - (overs * 6)
            wickets = 10 - wickets
            crr = score / overs
            rrr = runs_left / (balls_left / 6)
            input_data = pd.DataFrame({'BattingTeam': [batting_team], 'BowlingTeam': [bowling_team],
                                       'runs_left': [runs_left], 'balls_left': [balls_left],
                                       'wickets_remaining': [wickets], 'target': [target], 'crr': [crr], 'rrr': [rrr]})
            x = ct.transform(input_data)
            new_data_point = x.reshape(1, -1)
            probabilities = best_model.predict(new_data_point)
            pr1 = probabilities[0][0]
            pr2 = probabilities[0][1]
            st.header(batting_team + " = " + str(round(pr2 * 100)) + "%")
            st.header(bowling_team + " = " + str(round(pr1 * 100)) + "%")

    elif page == "Trophy Winners and Runner Ups":
        st.write("IPL Winners and Runner Ups list")
        st.write(final_match.reset_index(drop=True))
        # Visualization for Winners
        st.write("Frequency of IPL Wins Finishes by Team")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=final_match, y="Winner", order=final_match["Winner"].value_counts().index)
        plt.xlabel("Count")
        plt.ylabel("Teams")
        st.pyplot(plt)

        # Visualization for Runner-ups
        st.write("Frequency of IPL Runner-up Finishes by Team")
        plt.figure(figsize=(10, 6))
        sns.countplot(data=final_match, y="Runner", order=final_match["Runner"].value_counts().index)
        plt.xlabel("Count")
        plt.ylabel("Teams")
        st.pyplot(plt)

    elif page == "Team History":
        st.write("Team History")

        # Select team
        selected_team = st.selectbox("Select Team", teams_data['Team'].unique(), key="team_selector")

        # Filter data for the selected team
        team_data = teams_data[teams_data["Team"] == selected_team]

        # Display team statistics
        st.write("Total Matches Played:", team_data["Matches_Played"].sum())
        st.write("Total Wins:", team_data["Win"].sum())
        st.write("Total Losses:", team_data["Loss"].sum())
        st.write("Total Trophies:", len(final_match[final_match["Winner"] == selected_team]))
        st.write("Total Runner-up Finishes:", len(final_match[final_match["Runner"] == selected_team]))

        # Plot team performance over the years
        plt.figure(figsize=(10, 6))
        plt.plot(team_data["Season"], team_data["Win"], marker='o', label='Wins', color='green')
        plt.plot(team_data["Season"], team_data["Loss"], marker='o', label='Losses', color='red')
        plt.plot(team_data["Season"], team_data["Matches_Played"], marker='o', label='Matches Played', color='blue')
        plt.title(selected_team + ' Performance (2008-2022)')
        plt.xlabel('Year')
        plt.ylabel('Number of Matches')
        plt.xticks(team_data["Season"], rotation=45)
        plt.legend()
        st.pyplot(plt)

# Run the main function
if __name__ == "__main__":
    main()
