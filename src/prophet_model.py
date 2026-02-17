from prophet import Prophet

def run_prophet(df):

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=12, freq='ME')
    forecast = model.predict(future)

    return model, forecast
