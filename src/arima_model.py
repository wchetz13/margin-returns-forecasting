from statsmodels.tsa.arima.model import ARIMA

def run_arima(series, steps=30, order=(2,0,2)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()

    forecast_obj = model_fit.get_forecast(steps=steps)

    mean = forecast_obj.predicted_mean
    ci = forecast_obj.conf_int()

    return mean, ci
