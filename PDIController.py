class PDIController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, setpoint=0.5):
        self.Kp = Kp  # Ganancia proporcional
        self.Ki = Ki  # Ganancia integral
        self.Kd = Kd  # Ganancia derivativa
        self.setpoint = setpoint  # Valor objetivo (ej: 50% tasa de victoria)

        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, current_win_rate, dt=1.0):
        """
        Calcula el nuevo ajuste basándose en el valor actual del sistema.

        :param current_win_rate: tasa de victoria actual (ej: 0.5 para 50%)
        :param dt: tiempo entre llamadas (por defecto 1.0 si se llama cada iteración)
        :return: ajuste de la temperatura
        """
        error = self.setpoint - current_win_rate
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        adjustment = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.previous_error = error

        return adjustment