import dash
import dash_html_components as html
import dash_core_components as dcc


app = dash.Dash(__name__, 
                # external_stylesheets=external_stylesheets,
                meta_tags=[
                    # A description of the app, used by e.g.
                    # search engines when displaying search results.
                    {
                        'name': 'description',
                        'content': 'My description'
                    },
                    # A tag that tells Internet Explorer (IE)
                    # to use the latest renderer version available
                    # to that browser (e.g. Edge)
                    {
                        'http-equiv': 'X-UA-Compatible',
                        'content': 'IE=edge'
                    },
                    # A tag that tells the browser not to scale
                    # desktop widths to fit mobile screens.
                    # Sets the width of the viewport (browser)
                    # to the width of the device, and the zoom level
                    # (initial scale) to 1.
                    #
                    # Necessary for "true" mobile support.
                    {
                    'name': 'viewport',
                    'content': 'width=device-width, initial-scale=1.0'
                    }]
)

app.renderer = '''
var renderer = new DashRenderer({
    request_pre: (payload) => {
        // print out payload parameter
        console.log(payload);
    },
    request_post: (payload, response) => {
        // print out payload and response parameter
        console.log(payload);
        console.log(response);
    }
})
'''

app.layout = html.Div([html.H2('Simple Dash App'),
                      dcc.Markdown((""" ** Custom Dash Renderer **

                        You can tell your dash renderer to use custom meta tags defined at the dash instance.
                      """)),
                      dcc.Markdown((""" ** CSS Styling **

                      You don't have to do an external stylesheet like the one below:

                        `external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']`

                      You can also download the stylesheet that you want and  place it inside your assets folder and dash_renderer will automatically read it in. 
                      
                      If your Dash app receives a lot of traffic, you should  host the CSS somewhere else. One option is hosting it on GitHub's Gist and serving it through the free CDN RawGit.
                      """)),
])

if __name__ == '__main__':
    app.run_server(debug=True)