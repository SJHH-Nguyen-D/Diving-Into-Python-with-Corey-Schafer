import dash
import dash_core_components as dcc
import dash_html_components as html

# external JavaScript files
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]

# external CSS stylesheets
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

app = dash.Dash(__name__,
                external_scripts=external_scripts,
                external_stylesheets=external_stylesheets
)

app.layout = html.Div([
    html.Div(
        children=html.Div([
            html.H5('Overview'),
            html.Div('''
                This example makes use of the Image HTML tag as well as the use of external styling and javascript files which are read in before the local assets are.
            ''')
        ])
    ),
    html.Img(src="/assets/MHWI-Glavenus_Render_001.png"),
    html.Img(src="/assets/multipage_app_layout.png"),
])

if __name__ == '__main__':
    app.run_server(debug=True)