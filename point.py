class Point:
    # Point(2,3) -> x=2, y=3
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def get_x(self):
        return self._x
    
    def get_y(self):
        return self._y
    

    def align_point(self, a) -> None:
        if abs(self._x - a.get_x()) < 75:
            self._x = a.get_x()
        if abs(self._y - a.get_y()) < 75:
            self._y = a.get_y()

    