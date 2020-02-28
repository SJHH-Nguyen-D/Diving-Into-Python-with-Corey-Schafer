import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

layout = html.Div([
    html.H2('Employee Form Submission'),
    html.Br(),
    dcc.Markdown(("""
    
    Please fill out the employee characteristics intake form to recieve a prediction of employee performance score.
    
    """)),
    html.Br(),
    html.Div(children=[

        html.Div(children=[
        html.Label("First Name"),
        html.Div(),
        dcc.Input(id='firstname', type='text', value='John')
        ]),

        html.Div([html.Br()]),
        html.Div(children=[
        html.Label("Last Name"),
        html.Div(),
        dcc.Input(id='lastname', type='text', value='Doe')
        ]),

        html.Div([html.Br()]),

        html.Div([
        html.Label('Gender'),
        dcc.RadioItems(
            id='gender_r',
            options=[
                {'label': 'Male', 'value': "male"},
                {'label': 'Female', 'value': "female"},
                {'label': 'Other', 'value': 'other'},
                {'label': 'I prefer not to say', 'value': 'withheld'},
            ], value='male'),
        html.Div(id="output-gender")
        ]),

        html.Div([html.Br()]),

        html.Div([
        html.Label("Country of Birth"),
        dcc.Dropdown(
            id='cnt_brth',
            options=[
                {'label': 'Canada', 'value': 0}, # These values are just placeholder. we will look at getting the accurate mapping afterwards
                {'label': 'St. Lucia', 'value': 1}, 
                {'label': 'United States of America', 'value': 2},
            ], value='Canada')
        ]),
        
        html.Div([html.Br()]),

        html.Div([
        html.Label("Primary Language Spoken at Home"),
        dcc.Dropdown(
            id='lng_home',
            options=[
                {'label': 'English', 'value': 0}, # These values are just placeholder. we will look at getting the accurate mapping afterwards
                {'label': 'French', 'value': 1}, 
                {'label': 'Spanish', 'value': 2},
            ], value='English')
        ]),
        
        html.Div([html.Br()]),

        html.Div([
        html.Label("Primary language you used completed the intake questionnaire form"),
        dcc.Dropdown(
            id='lng_ci',
            options=[
                {'label': 'English', 'value': 0}, # These values are just placeholder. we will look at getting the accurate mapping afterwards
                {'label': 'French', 'value': 1}, 
                {'label': 'Spanish', 'value': 2},
                {'label': 'Danish', 'value': 3},
                {'label': 'Norwegian', 'value': 4},                
            ], value="English")
        ]),
        
        html.Div([html.Br()]),

        html.Div([
        html.Label("Current and historical work status"),
        dcc.Dropdown(
            id='v92',
            options=[
                {'label': 'Pupil, student', 'value': 0}, # These values are just placeholder. we will look at getting the accurate mapping afterwards
                {'label': 'Full-time employment', 'value': 1}, 
                {'label': 'Contractual employment', 'value': 2},
                {'label': 'Part-time employment', 'value': 3},            
            ], value='Full-time employment')
        ]),
        
        #########################################################################

        html.Div([html.Br()]),

        html.Div([
        html.Label("Estimated index measurement of use of influencing capabilities at place of employment (estimated from other intake questionnaire)  "),
        html.Div([html.Br()]),
        dcc.Input(id='influence', type='number', min=0, max=6.0, value=0, debounce=True)
        ]),
        
        html.Div([html.Br()]),

        html.Div([
        html.Label("Hours Spent at Job or Business a Week  "),
        html.Div([html.Br()]),
        dcc.Input(id='v1', type='number', min=0, max=168, step=1, value=0, debounce=True)
        ]),
        
        html.Div([html.Br()]),

        html.Div([
        html.Label("Number of years of paid worked in lifetime  "),
        dcc.Input(id='v97', type='number', min=0, max=100, step=1, value=0, debounce=True)
        ]),

        html.Div([html.Br()]),

        html.Div(id='display-score'),

        html.Div([html.Br()]),

    ]),

    html.Button('Submit', id='submit-values'),
    
])




# the input channels for the callback function are positional from the decorator and piping directly into the function parameters
@app.callback(
    Output(component_id='display-score', component_property='children'),
    [Input(component_id='firstname', component_property='value'),
    Input(component_id='lastname', component_property='value'),
    Input(component_id='gender_r', component_property='value'),
    Input(component_id='cnt_brth', component_property='value'),
    Input(component_id='lng_home', component_property='value'),
    Input(component_id='lng_ci', component_property='value'),
    Input(component_id='v92', component_property='value'),
    Input(component_id='influence', component_property='value'),
    Input(component_id='v1', component_property='value'),
    Input(component_id='v97', component_property='value')
    ]) # value is the property 
def display_employee_summary(firstname, lastname, gender, cntrybrth, home_language, exercise_language, employment_situation, influence, weekly_hours, lifetime_work_years):
    return ("Hello, my name is {} {}. I indentify as a {}. I was born in {}. I primarily speak {} at home. "\
        "I completed the exercise in {}. My current work situation can only be described as {}. My influence score is {}, I work about {} hours a week with an approximately {} years of paid work in my lifetime.").format(firstname, lastname, gender, cntrybrth, home_language, exercise_language, employment_situation, influence, weekly_hours, lifetime_work_years)

##########################################################################################
# the input channels for the callback function are positional from the decorator and piping directly into the function parameters
# In order for this function to work, you will need an intermediate function to recognize the encoded mapping for each categorical variable string from the read,
# and then convert it to the encoding scheme that is congruent with that of the original dataset so that the model can do inference on it; then you will have to turn it into an X_train np.array to be ingested by the predictive model.

# also, in order for the model to work, even might even have to retrain the exported pipeline again on the postprocessed data to get its weights.
# unforunately in my previous endeavors, I forgot to save the weights and coefficients of my learned model so we might have to revisit that old project again to get the weights.

# in the mean time, we can just instantiate the model and predict (after having loaded the the weights in the .npy file)

# @app.callback(
#     Output(component_id='display-score', component_property='children'),
#     [Input(component_id='firstname', component_property='value'),
#     Input(component_id='lastname', component_property='value'),
#     Input(component_id='gender_r', component_property='value'),
#     Input(component_id='cnt_brth', component_property='value'),
#     Input(component_id='lng_home', component_property='value'),
#     Input(component_id='lng_ci', component_property='value'),
#     Input(component_id='v92', component_property='value'),
#     Input(component_id='influence', component_property='value'),
#     Input(component_id='v1', component_property='value'),
#     Input(component_id='v97', component_property='value')
# def display_estimated_employee_performance_score(firstname, lastname, gender):
#     return "Hello {} {}. I indentify as a {}".format(firstname, lastname, gender)
########################################################################################

##########################################################################################

# FEATURES TO INCLUDE
# create a form, with a mapping for each
# include a fun support column for first and last name of an individual

# Attributes of the dataset... have to find the data set that had its features selected
# a few of these columns can be consolidated into a few options
#

# we want to also be able to have the prediction for your entry by passed as information to the visualization app (another page that does the visualization)
# and see how that person stacks up against your person.
"""
1. v1 = hours per week at job or business
    min=0, max=168
2. isco2c = Occupational classification of respondent's job at 2-digit level (ISCO 2008), current job (estimated)
    set of possible codes. Only include the codes that were part of the final dataset inclusion for only possible values
3. influence = Indexed measurement of use of influencing capabilities at place of employment (estimated)
    float from 0 to X
4. v97 = Current and historical work status - Years of paid work during lifetime (top-coded at 47)
    min=0, max=100 int
5. lng_home_deu = Language most often spoken in domestic setting - Respondent (ISO 639-2/T) (coded)
    what is the code for deutch?
6. lnh_home_hun = Language most often spoken in domestic setting - Respondent (ISO 639-2/T) (coded)
    what is the code for hungarian
7. cnt_brth_Saint Lucia = country of birth
8. lng_ci_dan = Language for exercise (estimated, ISO 639-2/T)
    what is the code for danish?
9. lng_ci_nor = Language for exercise (estimated, ISO 639-2/T)
    what is the code for norwegian
10. v92_Pupil, student = Current and historical work status - Subjective status

# what the form should look like
2 fields for name
8 fields to predict on
1 submission button
"""