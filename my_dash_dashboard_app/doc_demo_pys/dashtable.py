import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')

# first column is just indices for each row
df = df.iloc[:, 1:]

def generate_table(dataframe, max_rows=10):
    max_rows = 15
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# requests_pathname_prefix="/app2"

app.layout = html.Div(children=[
    html.H4(children='US Agriculture Exports (2011)'),
    dcc.Markdown((""" ** Static Table Generation with dash_html_components.tr **

    A static table is generated below using the generate_table function.
    """)),
    generate_table(df),
    dcc.Markdown((""" ** Dash-Tables **

    The below graphic is rendered using a dash-table as opposed to a function to generate the dash_html_components-based table, making it interactive and explorable.
    
    """)),
    dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict("rows"))
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)