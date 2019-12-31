# %%
import jupyterlab_dash
import dash
import dash_html_components as html

viewer = jupyterlab_dash.AppViewer()

app = dash.Dash(__name__)

app.layout = html.Div('Hello World')

viewer.show(app)

# %%
