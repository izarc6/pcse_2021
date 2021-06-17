#!/usr/bin/env python


import rospy
from tf_sim.msg import Float32Stamped

desired_val = 0.0
max_integ_error = 0.0

Integral = 0
e_anterior = 0

def exec_pid(error):

  global Kp, Kd, Ki, Integral, e_anterior, max_integ_error

  #TODO calcula la sortida del PID fent servir els parametres Kp, Kd, Ki i max_integ_error

  #u(t) = Kp*e(t) + Kd*(de/dt) + Ki * Integral_0_t(e(Tau)d(Tau))
  #e(t): Error en el instante actual

  # Termino proporcional
  P = Kp*error

  # Termino derivativo
  D = Kd * ((error-e_anterior)/T)

  # Termino integral (regla de integracion trapezoidal)
  Integral = Integral + error
  
  # Clamping segun max_integ_error
  if Integral >= max_integ_error:
    Integral = max_integ_error
  elif Integral <= max_integ_error*-1:
    Integral = max_integ_error * -1

  I = Integral * Ki

  output = P + I + D
  
  rospy.loginfo("P: %f | I: %f | D: %f", P, I, D)
  rospy.loginfo("Resultado PID: %f", output)

  # Pasamos el output a porcentaje

  # Actualizamos el error anterior para la siguiente iteracion
  e_anterior = error

  if output >= 100:
    output = 100
  elif output <= 0:
    output = 0

  rospy.loginfo("Resultado PID (normalizado): %f", output)

  return output

def callbackDesired(input_val):

  global desired_val

  desired_val = float(input_val.data)


def callbackMeasured(measured_val):

  global desired_val

  error = desired_val - measured_val.data
  
  rospy.loginfo("Valor deseado: %f | Error actual: %f", desired_val, error)

  ctrl_val = exec_pid(error)

  now = rospy.get_rostime()

  msg = Float32Stamped()
  msg.data = ctrl_val
  msg.header.stamp = now
  pub.publish(msg)
    
def PID():

  global pub
  
  global T
  global Kp
  global Kd
  global Ki
  global max_integ_error

  rospy.init_node('PID')

  freq = rospy.get_param('~frequency', 0.1)
  T = 1.0/freq

  Kp = rospy.get_param('~Kp', 0.0)
  Kd = rospy.get_param('~Kd', 0.0)
  Ki = rospy.get_param('~Ki', 0.0)
  max_integ_term = rospy.get_param('~max_integ_term', 0.0)

  if Ki > 0.0:
    max_integ_error = max_integ_term/Ki

  rospy.Subscriber("~desired_val", Float32Stamped, callbackDesired)
  rospy.Subscriber("~measured_val", Float32Stamped, callbackMeasured)

  pub = rospy.Publisher('~output', Float32Stamped, queue_size=10)

  rospy.loginfo("PID running")


  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()

if __name__ == '__main__':

  PID()
