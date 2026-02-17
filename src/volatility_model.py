from arch import arch_model

def run_garch(returns):

    model = arch_model(returns * 100, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")

    forecast = model_fit.forecast(horizon=12)

    return forecast
