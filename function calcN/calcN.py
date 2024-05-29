def calcN(f, z): 
    # f - фокусное расстояние
    # z - высота съемки
    f_30m = 0.01229
    z_30m = 30
    n_30m = 256
    return round((f / z) / (f_30m / z_30m) * n_30m)

print(calcN(0.01229, 60))