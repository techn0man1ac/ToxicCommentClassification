# Toxic Comment Classification System UI by "Team 16.6"

The app written with the help of **streamlit** provides a user-friendly interface to observe and try out functionality of the included BERT-based models for comment toxicity classification.

# Structure Description

The app's functionality is divided into the following files:
- `home.py`: Implements the interface of a home page. Includes the whole project description (based on the respective README) and a footer displaying the list of services used throughout the development.
- `about.py`: Implements the interface of a page for project team description. Includes the link to the project repository on **github** team's name and the list of members with the links to their **github** pages.
- `metrics.py`: Implements the interface of a page for a display of model metrics as well as other visualizations. Includes *accuracy*, *precision*, *recall* and confusion matrices of every toxicity class classification for a chosen model.
- `classify.py`: Implements the interface of a page with the classification functionality of the models. Includes model selection as well as text input (directly or through file upload) and a button that begins the classification process. After pressing the prediction of the chosen model is displayed below. If the box for "detailed toxicity" is checked, then a bar plot indicating toxicity classes that are present in the text is displayed additionally.

The following directories are necessary for the app:
- `data`: Contains *csv* files with metric confusion matrix values for each model.
- `imgs`: Contains the team logo as well as used services' logos.
- `saved_models`: Contains the trained models.