import streamlit as st


def about_page():

    # Link to project repository
    st.markdown('### [Team repository](https://github.com/techn0man1ac/ToxicCommentClassification)')

    # Team name
    st.title('Team 16.6:')

    # Team member list
    team = [
        {"name": "Serhii Trush", "role": "Team Lead", "github": "https://github.com/techn0man1ac"},
        {"name": "Oleksandr Kovalenko", "role": "SCRUM Master, Backend Developer", "github": "https://github.com/AlexandrSergeevichKovalenko"},
        {"name": "Olena Mishchenko", "role": "Data Scientist", "github": "https://github.com/Alena-Mishchenko"},
        {"name": "Ivan Shkvyr", "role": "Backend Developer", "github": "https://github.com/IvanShkvyr"},
        {"name": "Oleksii Yeromenko", "role": "Frontend Developer", "github": "https://github.com/oleksii-yer"},
        {"name": "Polina Mamchur", "role": "Creative Director", "github": "https://github.com/polinamamchur"}
    ]

    # Display team members in columns
    for i in range(0, len(team), 2):  # Display 2 members per row
        cols = st.columns(2)
        for col, member in zip(cols, team[i:i+2]):
            with col:
                # Display member name and role
                st.markdown(f"### [{member['name']}]({member['github']})")
                st.markdown(f"*{member['role']}*")
                