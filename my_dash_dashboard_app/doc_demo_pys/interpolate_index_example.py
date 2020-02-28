import dash
import dash_html_components as html
import dash_core_components as dcc
from pprint import PrettyPrinter


class CustomDash(dash.Dash):
    def interpolate_index(self, **kwargs):
        # Inspect the arguments by printing them
        pp = PrettyPrinter(indent=4)
        pp.pprint(kwargs)
        return '''
        <!DOCTYPE html>
        <html>
            <head>
                <title>My App</title>
            </head>
            <body>
                <img src='https://share.bannersnack.com/bdtjazvw5/' alt='site banner'>
                {app_entry}
                {config}
                {scripts}
                {renderer}
            </body>
        </html>
        '''.format(
            app_entry=kwargs['app_entry'], # you can supply your own arguments here
            config=kwargs['config'],
            scripts=kwargs['scripts'],
            renderer=kwargs['renderer'])

app = CustomDash()

# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div([
    dcc.Markdown(("""
    ** Simple Dash App **

    This class provides a template for html generation customized from the default html template that is rendered by Dash and React.js

    The header and footer of this page are fixed as templates that will be generated for each page stemming from this index template.
    """))
    ])

if __name__ == '__main__':
    app.run_server(debug=True)