import plotly.express as px
data_canada = px.data.gapminder().query("country == 'Australia'")
fig = px.bar(data_canada, x='year', y='pop')
fig.show()
