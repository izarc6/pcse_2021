#!/usr/bin/env python

import rospy
from tf_sim.msg import Float32Stamped

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist

import tf
import math

PO = 0.0  # Punto Objetivo
PA = 0.0  # Punto Actual
max_integ_error_l = 0.0
max_integ_error_a = 0.0

# PID lineal
Kp_l = 0.0
Kd_l = 0.0
Ki_l = 0.0
Integral_l = 0.0
e_anterior_l = 0.0

V_lin = 0.0

# PID angular
Kp_a = 0.0
Kd_a = 0.0
Ki_a = 0.0
Integral_a = 0.0
e_anterior_a = 0.0

V_ang = 0.0

# Error lineal es la distancia euclidea entre PO y PA
# Determina si el Kobuki debe acelerar
def exec_pid_lineal(error_lineal):

  global Kp_l, Kd_l, Ki_l, Integral_l, e_anterior_l, max_integ_error_l, V_lin

  # Termino proporcional
  P = Kp_l*error_lineal

  # Termino derivativo
  D = Kd_l * V_lin

  # Termino integral (regla de integracion trapezoidal)
  Integral_l = Integral_l + error_lineal
  
  # Clamping segun max_integ_error
  if Integral_l >= max_integ_error_l:
    Integral_l = max_integ_error_l
  elif Integral_l <= max_integ_error_l*-1:
    Integral_l = max_integ_error_l * -1

  I = Integral_l * Ki_l

  output = P - D + I
  
  #rospy.loginfo("P: %f | I: %f | D: %f", P, I, D)
  rospy.loginfo("Resultado PID (lineal): %f", output)

  # Pasamos el output a porcentaje

  # Actualizamos el error anterior para la siguiente iteracion
  e_anterior_l = error_lineal

  return output

# Error angular es el angulo entre el Kobuki y el PO
# Determina si el robot debe girarse y de cuanto
def exec_pid_angular(error_angular):
  global Kp_a, Kd_a, Ki_a, Integral_a, e_anterior_a, max_integ_error_a, V_ang

  # Termino proporcional
  P = Kp_a*error_angular

  # Termino derivativo
  D = Kd_a * V_ang

  # Termino integral (regla de integracion trapezoidal)
  Integral_a = Integral_a + error_angular
  
  # Clamping segun max_integ_error
  if Integral_a >= max_integ_error_a:
    Integral_a = max_integ_error_a
  elif Integral_a <= max_integ_error_a*-1:
    Integral_a = max_integ_error_a * -1

  I = Integral_a * Ki_a

  output = P - D + I
  
  #rospy.loginfo("P: %f | I: %f | D: %f", P, I, D)
  rospy.loginfo("Resultado PID (angular): %f", output)

  # Pasamos el output a porcentaje

  # Actualizamos el error anterior para la siguiente iteracion
  e_anterior_a = error_angular

  return output


# Algoritmo del enunciado, posicion actual
def callbackOdom(PA):
  global PO, V_lin, V_ang

  # Inicializacion velocidades lineal y angular
  C_lin = 0
  C_ang = 0

  # Calculo del error
  Ex = PO.position.x - PA.pose.pose.position.x
  Ey = PO.position.y - PA.pose.pose.position.y

  De = distanciaEuclidea(PO, PA)

  # Si estamos a mas de 5 cm
  if De > 0.05:
    yaw = orientacion(PA) # Apendix A, orientacion
    theta = errorAngular(PO, PA, yaw, De) # Apendix B, error angular

    # Actualizacion velocidades segun datos odometricos
    V_lin = PA.twist.twist.linear.x
    V_ang = PA.twist.twist.angular.z

    C_ang = exec_pid_angular(theta)

    # Si el error angular es de mas de 10 grados
    if abs(math.degrees(theta)) < 10:
      C_lin = exec_pid_lineal(De)

  # Publicamos las velocidades
  msg = Twist()
  msg.linear.x = C_lin
  msg.angular.z = C_ang
  pub_twist.publish(msg)


# Actualiza la posicion objetivo
def callbackPose(pos_objetivo):
  global PO # Punto objetivo
  PO = pos_objetivo

# Calcula la distancia euclidea entre los dos puntos
# En realidad es simplemente hacer Pitagoras
def distanciaEuclidea(PO, PA):
  x_o = PO.position.x
  x_a = PA.pose.pose.position.x
  y_o = PO.position.y
  y_a = PA.pose.pose.position.y

  return math.sqrt(math.pow((x_o - x_a), 2) + math.pow((y_o - y_a), 2))

# Apendix A
def orientacion(measured_odom):
  quaternion = (measured_odom.pose.pose.orientation.x, measured_odom.pose.pose.orientation.y, measured_odom.pose.pose.orientation.z, measured_odom.pose.pose.orientation.w)
  euler = tf.transformations.euler_from_quaternion(quaternion)

  return euler[2] # Yaw

# Apendix B
def errorAngular(PO, PA, yaw, De):
  Ex = PO.position.x - PA.pose.pose.position.x
  Ey = PO.position.y - PA.pose.pose.position.y

  # Angulo
  theta = math.acos( ((math.cos(yaw) * Ex) + (math.sin(yaw) * Ey)) / De )

  # Permite saber si es a la izquierda o a la derecha
  delta = (math.cos(yaw) * Ey) - (math.sin(yaw) * Ex)

  if delta < 0:
    theta *= -1

  return theta

    
def PID():

  global pub, pub_twist
  
  global T
  global Kp_l, Kp_a
  global Kd_l, Kd_a
  global Ki_l, Ki_a
  global max_integ_error_l, max_integ_error_a

  rospy.init_node('PID')

  freq = rospy.get_param('~frequency', 0.1)
  T = 1.0/freq

  Kp_l = rospy.get_param('~Kp_l', 0.0)
  Kd_l = rospy.get_param('~Kd_l', 0.0)
  Ki_l = rospy.get_param('~Ki_l', 0.0)
  max_integ_term_l = rospy.get_param('~max_integ_term_l', 0.0)

  if Ki_l > 0.0:
    max_integ_error_l = max_integ_term_l/Ki_l

  Kp_a = rospy.get_param('~Kp_a', 0.0)
  Kd_a = rospy.get_param('~Kd_a', 0.0)
  Ki_a = rospy.get_param('~Ki_a', 0.0)
  max_integ_term_a = rospy.get_param('~max_integ_term_a', 0.0)

  if Ki_a > 0.0:
    max_integ_error_a = max_integ_term_a/Ki_a


  rospy.Subscriber("~current_point", Odometry, callbackOdom)
  rospy.Subscriber("~desired_point", Pose, callbackPose)

  pub_twist = rospy.Publisher("~output_twist", Twist, queue_size=10)

  rospy.loginfo("PID running")

  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()

if __name__ == '__main__':

  PID()
