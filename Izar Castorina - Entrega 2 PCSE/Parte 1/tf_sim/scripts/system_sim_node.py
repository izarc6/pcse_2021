#!/usr/bin/env python

# Izar Castorina - Entrega 1 Practica 2 P.C.S.E.

import rospy
from tf_sim.msg import Float32Stamped

k1 = 0
y_k_1 = 0
y_k_2 = 0
x_k_1 = 0
x_k_2 = 0

k2 = 0
f_k_1 = 0
f_k_2 = 0

df1 = 0 # y_k (FDI)
df2_y = 0 # y_k (FDII)
df2_f = 0 # f_k (FDII)

def direct_form_I(x_k):

  global y_k_1, y_k_2, x_k_1, x_k_2, df1

  # TO_DO Implementar el codi a partir de les equacions
  # en diferencies obtingudes amb la forma directa I

  #v(3) = k-2, v(2) = k-1, v(1) = k

  #vx(3) = vx(2), vx(2) = vx(1), vx(1) = x_k
  #vy(3) = vy(2), vy(2) = vy(1)
  #vy(1) = 1.9123*vy(2) - 0.9203*vy(3) + 0.0797*vx(2) - 0.0717*vx(3)

  #y_k = 0

  x_k_2 = x_k_1
  x_k_1 = x_k
  y_k_2 = y_k_1

  y_k_1 = df1

  y_k = 1.9123*y_k_1 - 0.9203*y_k_2 + 0.0797*x_k_1 - 0.0717*x_k_2
  df1 = y_k

  rospy.loginfo('Resultado FD1: %f', y_k)


  return y_k

def direct_form_II(x_k):

  # TO_DO Implementar el codi a partir de les equacions
  # en diferencies obtingudes amb la forma directa II

  # vf(2) = f_k_1, vf(3) = f_k_2

  global f_k_1, f_k_2, df2_y, df2_f

  #f_k = 0
  #y_k = 0

  f_k_2 = f_k_1

  f_k_1 = df2_f

  f_k = x_k + 1.9123*f_k_1 - 0.9203*f_k_2
  df2_f = f_k
  y_k = 0.0797*f_k_1 - 0.0717*f_k_2
  df2_y = y_k

  rospy.loginfo('Resultado FD2: %f', y_k)

  return y_k


def callback(input_val):

  y1 = direct_form_I(input_val.data)
  y2 = direct_form_II(input_val.data)

  now = rospy.get_rostime()

  msg1 = Float32Stamped()
  msg1.data = y1
  msg1.header.stamp = now
  pub1.publish(msg1)

  msg2 = Float32Stamped()
  msg2.data = y2
  msg2.header.stamp = now
  pub2.publish(msg2)

    
def system():

  global pub1
  global pub2

  rospy.init_node('system')

  rospy.Subscriber("~input", Float32Stamped, callback)

  pub1 = rospy.Publisher('output_val_1', Float32Stamped, queue_size=10)
  pub2 = rospy.Publisher('output_val_2', Float32Stamped, queue_size=10)

  rospy.loginfo("System running")


  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()

if __name__ == '__main__':

  system()
