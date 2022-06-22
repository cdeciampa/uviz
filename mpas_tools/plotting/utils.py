

class Colorbars:
    def __init__(self, name):
        if self.name == 'nws precip':
            
        self.nws_precip = [
            '#ffffff',  # 0 inches
            "#04e9e7",  # 0.01 - 0.10 inches
            "#019ff4",  # 0.10 - 0.25 inches
            "#0300f4",  # 0.25 - 0.50 inches
            "#02fd02",  # 0.50 - 0.75 inches
            "#01c501",  # 0.75 - 1.00 inches
            "#008e00",  # 1.00 - 1.50 inches
            "#fdf802",  # 1.50 - 2.00 inches
            "#e5bc00",  # 2.00 - 2.50 inches
            "#fd9500",  # 2.50 - 3.00 inches
            "#fd0000",  # 3.00 - 4.00 inches
            "#d40000",  # 4.00 - 5.00 inches
            "#bc0000",  # 5.00 - 6.00 inches
            "#f800fd",  # 6.00 - 8.00 inches
            "#9854c6",  # 8.00 - 10.00 inches
            "#fdfdfd"   # 10.00+
        ]